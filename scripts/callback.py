from typing import List

from ray import tune
from ray.tune.checkpoint_manager import Checkpoint
from ray.tune.trial import Trial


class SaveModel(tune.Callback):
    def on_checkpoint(self, iteration: int, trials: List[Trial],
                      trial: Trial, checkpoint: Checkpoint, **info):
        print(type(checkpoint))
        print(checkpoint)
        print(checkpoint.__dict__)
