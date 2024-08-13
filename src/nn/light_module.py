import torch
import logging

import numpy as np
import pytorch_lightning as L

from torch import nn
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from src.base.nn.loss import Metric
from src.nn.model import TabularModel
from src.nn.dataset import TrainDataset

class TablePredictor(L.LightningModule):
    def __init__(self, config, criterion: nn.Module, metric: Metric):
        super().__init__()
        
        self.criterion = criterion
        self.metric = metric
        self.config = config
                        
        self.step_outputs = {
            'val': [],
        }
        self.history: Dict[str, list[float]] = {
            'loss': [],
            metric.name: []
        }
        self.save_hyperparameters()
        self.init_console_logger()

    def configure_model(self):
        self.tabular_model = TabularModel(config=self.config)            
         
    def init_console_logger(self) -> None:
        self.console_logger = logging.getLogger()
        self.message_logging: str = (
            "Epoch: {epoch} Val Loss: {val_loss:.5f} Val {metric_name}: {metric_value:.5f}"
        )

    def training_step(self, batch, batch_idx):
        input_, labels = batch

        pred = self.forward(input_)
        loss = self.criterion(pred, labels)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_, labels = batch

        pred = self.forward(input_)
        
        loss = self.criterion(pred, labels)
        self.step_outputs['val'].append(
            {'loss': loss, 'pred': torch.sigmoid(pred), 'labels': labels}
        )
                    
    def on_validation_epoch_end(self):
        if self.trainer.global_step>0:
            self.__eval_metric()
    
    def on_validation_end(self):     
        if self.trainer.global_step>0:
            if self.trainer.is_global_zero:
                metrics = self.trainer.callback_metrics
                
                self.console_logger.info(
                    self.message_logging.format(
                        epoch=self.current_epoch,
                        val_loss=metrics['val_loss'],
                        metric_name=self.metric.name,
                        metric_value=metrics[f'val_{self.metric.name}']
                    )
                )
    
    def __eval_metric(self):
        loss = [out['loss'].reshape(1) for out in self.step_outputs['val']]
        loss = torch.mean(torch.cat(loss))
        
        preds = torch.cat(
            [out['pred'] for out in self.step_outputs['val']]
        )
        
        labels = torch.cat(
            [out['labels'] for out in self.step_outputs['val']]
        )
    
        metric_score = self.metric(y_true=labels, y_pred=preds)
        #initialize performance output
        res_dict = {
            f'val_loss': loss,
            f'val_{self.metric.name}': metric_score
        }
        self.history['loss'].append(metric_score)
        self.history[self.metric.name].append(metric_score)
        
        if self.trainer.sanity_checking:
            pass
        else:
            self.log_dict(res_dict, sync_dist= True)
            
        #free memory
        self.step_outputs['val'].clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def forward(self, inputs):
        output = self.tabular_model(inputs)
        
        return output
    
    def predict_step(self, batch, batch_idx):
        pred = self.forward(batch)
        
        return pred
    
class TabularDataModule(L.LightningDataModule):
    def __init__(
            self, 
            config: Dict[str, Any],
            train_matrix: Tuple[np.ndarray], valid_matrix: Tuple[np.ndarray], 
            cat_features_idxs: list[int]
        ):
        super().__init__()
        
        self.config: Dict[str, Any] = config
        self.train_matrix: Tuple[np.ndarray] = train_matrix
        self.valid_matrix: Tuple[np.ndarray] = valid_matrix
        
        self.cat_features_idxs: list[int] = cat_features_idxs
        
    def train_dataloader(self):
        train_dataset = TrainDataset(
            feature=self.train_matrix[0],
            target=self.train_matrix[1],
            cat_features_idxs=self.cat_features_idxs,
            config=self.config, inference=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.config['num_workers'],
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = TrainDataset(
            feature=self.valid_matrix[0],
            target=self.valid_matrix[1],
            cat_features_idxs=self.cat_features_idxs,
            config=self.config, inference=False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )
        return valid_loader
