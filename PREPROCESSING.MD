# Preprocessing[OPTIONAL]

### REQUIREMENTS:
Input dataset must be possible to read and convert to pandas, Files live csv, excel, numpy arrays and others plain datasets

This step is needed only initial preprocessing such as set splitting, transformations and feature filtering is needed

#### 1. Download example dataset from here
[csv dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) <br/>
[dataset details](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names) <br/>
[reference](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) <br/>
#### 2. Put dataset csv file to **path_to_project/data** folder with name _dataset.csv_

#### 3. Create config file like "preprocessing_<something>.yml" inside **configs** folder <br/>
Example: example_config_files/preprocessing_example.yml
In preprocessing config you specify reader(csv, xlsm, xlsx or other custom)

### 4. `python run_preprocessing --config-yml configs/preprocessing_<something>.yml`
The resulted data will be saved in **processed_data/**_dataset.csv_
