import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, Any
from itertools import product

class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(
            input_dim, hidden_dim
        )
        self.w2 = nn.Linear(
            input_dim, hidden_dim
        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = F.silu(self.w1(x)) * self.w2(x)
        return x
        
class EmbeddingGenerator(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(EmbeddingGenerator, self).__init__()
        
        self.config = config
        
        self.embedding_list = nn.ModuleList()
        self.num_categorical: int = len(self.config['cat_features_idxs'])
        assert len(self.config['cat_dim']) == self.num_categorical
        
        for cat_dim in self.config['cat_dim']:
            self.embedding_list.append(
                torch.nn.Embedding(cat_dim, 1)
            )

    def forward(self, categorical_tensor: torch.Tensor) -> torch.Tensor:
        post_embedding_output = torch.cat(
            [
                self.embedding_list[position](categorical_tensor[:, position])
                for position in range(self.num_categorical)
            ],
            dim=1
        )
        return post_embedding_output

class TabularModel(nn.Module):
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularModel, self).__init__()
        
        self.config = config
        self.num_features = config['num_features']
        self.has_numerical: bool = (self.num_features>len(self.config['cat_features_idxs']))
        
        self.__init_model()
        
        if len(self.config['cat_features_idxs']) > 0:
            self.has_categorical: bool = True
            
            self.embedding_generator = EmbeddingGenerator(
                config=config
            )
        
    def __init_model(self):
        self.feedforward_list: list[nn.Module] = []
        
        #add initial layer
        input_dim_list = [self.num_features] + self.config['hidden_dim_list'][:-1]
        hidden_dim_list = self.config['hidden_dim_list']

        for input_dim, hidden_dim in zip(input_dim_list, hidden_dim_list):
            self.feedforward_list.append(
                FeedForward(
                    input_dim=input_dim, 
                    hidden_dim=hidden_dim
                )
            )
        self.classifier = nn.Linear(
            hidden_dim_list[-1], 
            self.config['num_labels'],
        )

    def forward(self, inputs):
        input_network: list = []

        if self.has_numerical:
            input_network.append(inputs[0])
            
        if self.has_categorical:
            input_network.append(
                self.embedding_generator(inputs[1])
            )
        
        x = torch.cat(input_network, dim=1)

        for ff in self.feedforward_list:
            x = ff(x)
            
        output = self.classifier(x)
        
        return output