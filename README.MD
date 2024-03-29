# Simple ML Pipeline

Python framework to speed up the ML process starting from dataset collection and preprocessing to training and prediction

## Installation

git clone https://gitlab.g42.ae/uplatform/experimentation-ml-pipeline.git
pip install -r requirements.txt
## Usage

All configs are intended to be tweaked for appropriate model/dataset


### Example usage:
**If you already have a torch.Dataset skip preprocessing** <br />
[PREPROCESSING](PREPROCESSING.MD)
```terminal
# collects and transforms the data
python run_preprocessing.py --config-yml=configs/preprocessing.yml
```

[TRAINING](TRAINING.md)
```terminal
# trains one or multimple models 
python run_train.py --config-yml=configs/dense_net.yml
```

[PREDICTION](PREDICTION.md)
```terminal
# makes predictions with several models
python run_prediction.py --config-yml=configs/prediction.yml
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.