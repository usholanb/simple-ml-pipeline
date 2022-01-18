# Training 

##### Description:
1. Trains a model specified in config file 
2. Saves classifier and best checkpoints 
3. Saves logs for tensorboard

## Step 1: Dataset 
### OPTION 1 - using preprocessed files(from [PREPROCESSING](PREPROCESSING.MD) STEP)


#### 1. Create train config in **configs** like train_<something>.yml
Example: **example_config_files/**_train_dense_net_example.yml_
In the train config you can choose model, dataset, trainer, metrics, dataloader configs, features see the example
##### Requirement: Add these two parameters under special_inputs, more details in example config
special_inputs: <br/>
input_dim: 53  
n_outputs: 2


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
4. Original dataset and code [reference](https://github.com/alexmonti19/dagnet)

#### 2. Insert the appropriate torch.Dataset class in **path_to_project/modules/datasets/**
The torch.Dataset for this specific example is already inserted
For reference see **path_to_project/modules/datasets/basket_dataset.py**

#### 3. Create and insert training config in **path_to_project/configs** folder
Use example **example_config_files/**_train_dagnet_example.yml_

#### 4. `python run_train --config-yml configs/train_<something>.yml`


## Step 2: Model
#### 1. Take a look at one of the models already created in modules/models folder. You already can use them or any of sklearn models. In addition, you can create your own by inheriting BaseModel or DefaultModel. Just override the abstract methods.
#### 2. Don't forget to register it line other models(put "@registry.register_model('**my_custom_model**')" above your model class declaration). In this case **my_custom_model** will be the name of your model that you specify in your train and prediction config files.
#### 3. You have to specify tag of your model in config file(see examples). This way models of the same type(for example: 2 **xgboost**s) will be distinguished. They will be named with model class name and tag name in underscore style.
#### 4. Once the model is written make sure it's in modules/models folder and run
`python run_train --config-yml configs/train_<something>.yml`


#### Tips:
1. For binary classification torch models use **n_outputs: 1**
2. For all classification torch models beware you need to specify self.output_function (last layer activation) and self.prediction_function (used during the prediction step, usually nn.Softmax) yourself in your model class.
