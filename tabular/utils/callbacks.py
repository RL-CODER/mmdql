from copy import deepcopy
import numpy as np
from mushroom_rl.utils.table import EnsembleTable

class CollectQs:
    def __init__(self, approximator):
        self._approximator = approximator
        self._qs = list()

    def __call__(self, **kwargs):
        if isinstance(self._approximator, EnsembleTable):
            qs = list()
            for m in self._approximator.model:
                qs.append(m.table)
            self._qs.append(deepcopy(qs))
        else:
            self._qs.append(deepcopy(self._approximator.table))

    def get_values(self):
        return self._qs