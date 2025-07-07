from mushroom_rl.utils.table import Table
import numpy as np

class Parameter(object):
    def __init__(self, value, min_value=None, size=(1,)):
        self._initial_value = value
        self._min_value = min_value
        self._n_updates = Table(size)

    def __call__(self, *idx, **kwargs):
        if self._n_updates.table.size == 1:
            idx = list()
        self.update(*idx, **kwargs)

        return self.get_value(*idx, **kwargs)

    def get_value(self, *idx, **kwargs):
        new_value = self._compute(*idx, **kwargs)
        if self._min_value is None or new_value >= self._min_value:
            return new_value
        else:
            return self._min_value

    def _compute(self, *idx, **kwargs):
        return self._initial_value

    def update(self, *idx, **kwargs):
        self._n_updates[idx] += 1

    @property
    def shape(self):
        return self._n_updates.table.shape

class LogarithmicDecayParameter(Parameter):
    def __init__(self, value, C=1., min_value=None, size=(1,)):
        self._C = C
        super(LogarithmicDecayParameter, self).__init__(value, min_value, size)

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)
        lr = 1 - np.e ** (-(1 / (n+1) * (self._C + 2 * np.log(n + 1))))
        return lr