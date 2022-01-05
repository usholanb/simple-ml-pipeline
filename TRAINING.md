# Training 

### OPTION 1 - using preprocessed files(from [PREPROCESSING](PREPROCESSING.MD) STEP)


#### 1. Create train config in **configs** like train_<something>.yml
Example: **example_config_files/**_train_dense_net_example.yml_
In the train config you can choose model, dataset, trainer, metrics, dataloader configs, features see the example

#### 2. `python run_train --config-yml configs/train_<something>.yml`
**a.** The results will be pickle file of classifier saved in **classifiers** <br />
**b.** If training torch model the epoch loss and other metrics specified in the model itself will be logged for tensorboard in **train_results**<br />

#### 3. Repeat steps 1 and 2 but with other examples
**a.** Logistic Regression config **example_config_files/**_train_logistic_regression_example.yml_
###### Note changes in Logistic Regression:
&emsp; 1. Trainer changed to sklearn_trainer <br />
&emsp; 2. model.name, model.tag changed <br />
&emsp; 3. special inputs are accords with inputs of sklearn.linear_model.LogisticRegression class<br />
&emsp; 4. dataset.Dataloader was removed as it's not needed<br />
**b.** Logistic Regression config **example_config_files/**_train_xgboost_example.yml_

#### 4. The result classifiers are saved in classifiers folder
If trained torch model the epoch wise and additional metrics are saved in train_results in **tensorboard** format


### OPTION 2 - if you have your own torch.Dataset
#### Example:
#### 1. Prepare data
1. You must be in G42 network to access this dataset
2. Download [dataset](https://uan-example-datasets.obs.ae-ad-1.g42cloud.com/dagnet.zip) of x y coordinates of players in NBA games
3. Unzip it and put in **data** folder

#### 2. Insert the appropriate torch.Dataset class in **path_to_project/modules/datasets/**
The torch.Dataset for this specific example is already inserted
For reference see **path_to_project/modules/datasets/basket_dataset.py**

#### 3. Create and insert training config in **path_to_project/configs** folder
Use example **example_config_files/**_train_dagnet_example.yml_

#### 4. `python run_train --config-yml configs/train_<something>.yml`
