# Prediction
#####Description:
1. Runs all specified models and runs against the specified dataset  
2. Saves the results of the run
3. Can be used to create graphs

#### 1. Create train config in **configs** like prediction_<something>.yml
Example: **example_config_files/**_prediction_example.yml_
In the train config you can choose models, dataset and metrics

#### 2. `python run_train --config-yml configs/prediction_<something>.yml`
The results will be files with model/metrics saved in **path_to_project/predictions** folder <br\> 






