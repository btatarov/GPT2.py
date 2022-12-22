import json
from operator import itemgetter
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


class Log:
    LOG_FILE = './log.json'
    PLOT_DIR = './plot/'

    def __init__(self, rebuild: Optional[bool] = False) -> None:
        self.load(rebuild)

    def __len__(self) -> int:
        return len(self._data)

    def load(self, rebuild: bool) -> None:
        if os.path.exists(self.LOG_FILE) and not rebuild:
            with open(self.LOG_FILE, 'r') as f:
                self._data = json.load(f)
        else:
            self._data = []

    def save(self) -> None:
        with open(self.LOG_FILE, 'w') as f:
            json.dump(self._data, f, indent=2)

    def add_epoch(self, epoch_time: float, logs: dict[str, float]) -> None:
        data = logs.copy()
        data['epoch'] = len(self) + 1
        data['time'] = epoch_time
        self._data.append(data)
        self.save()

    def plot(self) -> None:
        if len(self) < 1:
            raise RuntimeError('log file must have at least one epoch record')

        if not os.path.exists(self.PLOT_DIR):
            os.makedirs(self.PLOT_DIR)

        metrics = {
            'loss': 'loss',
            'acc': 'accuracy',
            'ent': 'crossentropy',
        }
        epochs = range(1, len(self) + 1)
        for metric, metric_name in metrics.items():
            plt.cla()
            plt.clf()
            plt.plot(epochs, list(map(itemgetter(metric), self._data)), color='darkblue', label=metric)
            plt.plot(epochs, list(map(itemgetter(f'val_{metric}'), self._data)), color='orange', label=f'val_{metric}')
            plt.title(metric_name.capitalize())
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(Path(self.PLOT_DIR) / f'{metric_name}.png', dpi=300)
