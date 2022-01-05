# Training


### 1. Create train config in **configs** like train_<something>.yml
#### Example: **example_config_files/**_train_dense_net_example.yml_
#### In the train config you can choose model, dataset, trainer, metrics, dataloader configs, features see the example

### 2. `python run_train --config-yml configs/train_<something>.yml`
#### a. The results will be pickle file of classifier saved in **classifiers** <br\>
#### b. If training torch model the epoch loss and other metrics specified in the model itself will be logged for tensorboard in **train_results**

### 3. Repeat steps 1 and 2 but with other examples
#### a. Logistic Regression config **example_config_files/**_train_logistic_regression_example.yml_
##### Note changes in Logistic Regression:
##### 1. Trainer changed to sklearn_trainer
##### 2. model.name, model.tag changed
##### 3. special inputs are accords with inputs of sklearn.linear_model.LogisticRegression class
##### 4. dataset.Dataloader was removed as it's not needed
#### b. Logistic Regression config **example_config_files/**_train_xgboost_example.yml_

### 4. The result classifiers are saved in classifiers folder
### If trained torch model the epoch wise and additional metrics are saved in train_results in **tensorboard** format






