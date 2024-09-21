# Set up data

Then unzip inside folder original_data

Create the required environment by executing following command:
```
//create venv
python -m venv .venv

//activate .venv
source .venv\Scripts\activate

//upgrade pip
python -m pip install --upgrade pip

//instal package in editable mode
python -m pip install -e .

//clean egg-info artifact
python setup.py clean
```

Download data for each state from https://www.census.gov/quickfacts/fact/table/{STATE}/PST045223
```
python concat_data.py
command.ps1
python create_additional_data.py
python create_economics_dataset.py
```