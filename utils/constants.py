import os

PROJECT_DIR = f'{os.path.dirname(os.path.abspath(__file__))}/..'

# call variable "*_DIR" if you want it to be created
# automatically if not exists. see function setup_directories()
DATA_DIR = f'{PROJECT_DIR}/data'
CONFIGS_DIR = f'{PROJECT_DIR}/configs'
TRAIN_RESULTS_DIR = f'{PROJECT_DIR}/train_results'
CLASSIFIERS_DIR = f'{PROJECT_DIR}/classifiers'
PREDICTIONS_DIR = f'{PROJECT_DIR}/predictions'
PROCESSED_DATA_DIR = f'{PROJECT_DIR}/processed_data'
MODULES_DIR = f'{PROJECT_DIR}/modules'
FOLDERS_NAMES = [
    'datasets',
    'losses',
    'metrics',
    'models',
    'predictors',
    'readers',
    'trainers',
    'transformers',
    'wrappers',
]


