from typing import List, AnyStr

import numpy as np
import nltk
import pandas as pd

from modules.transformers.base_transformers.base_transformer import BaseTransformer
from utils.registry import registry


def get_similar_words(words: List[AnyStr], coeff: float) -> None:
    """ Prints out similar works  """
    scores = {}
    for i, ww1 in enumerate(words[:-1]):
        for j in range(i + 1, len(words)):
            ww2 = words[j]
            length = (len(ww1) + len(ww2))
            score = (length - nltk.edit_distance(ww1, ww2)) * 1.0 / length
            if score > coeff:
                scores[(i, j)] = score

    for (i, j), score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        w1, w2 = words[i], words[j]
        print(f'"{w1}" ---- "{w2}", score: {score}\n')


@registry.register_transformer('ohe')
class OheTransformer(BaseTransformer):
    def apply(self, vector: np.ndarray) -> np.ndarray:
        nltk.download('wordnet')
        uniq = sorted(np.unique(vector))
        if isinstance(uniq[0], str):
            get_similar_words(uniq, coeff=0.8)
        val_to_index = {val: index for index, val in enumerate(uniq)}
        output = np.zeros((len(vector), len(val_to_index)))
        idx = np.array([val_to_index[val] for val in vector.tolist()])
        output[np.arange(len(vector)), idx] = 1
        return pd.DataFrame(data=output, columns=uniq)

