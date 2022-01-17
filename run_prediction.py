"""
This script uses the models specified in the config and gets the predictions
on the specified in the config dataset

The predictions and probabilities are saved in the prediction files in
predictions/ folder
"""
from typing import AnyStr, Dict
from modules.predictors.simple_predictor import SimplePredictor
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry


def get_predictor(k_fold_tag: AnyStr) -> SimplePredictor:
    setup_directories()
    from utils.flags import all_flags
    parser = all_flags['prediction'].parser
    args = parser.parse_args()
    configs = build_config(args)
    configs['dataset']['k_fold_tag'] = k_fold_tag
    setup_imports()
    predictor_name = configs.get('predictor', 'predictor')
    return registry.get_predictor_class(predictor_name)(configs)


def save_files(predictor: SimplePredictor, preds_ys: Dict) -> None:
    """ Prepares Dataset """
    predictor.save_metrics(preds_ys)
    predictor.save_predictions()
    predictor.save_graphs(preds_ys)


def run_prediction(k_fold_tag: AnyStr = '') -> None:
    print('started run_prediction')
    predictor = get_predictor(k_fold_tag)
    preds_ys = predictor.get_preds_ys()
    save_files(predictor, preds_ys)


if __name__ == '__main__':
    run_prediction()
