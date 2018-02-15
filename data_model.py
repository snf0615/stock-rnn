import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())

class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_percent=0.1,
                 normalized=True,
                 close_price_only=True):
        self.stock_sym = stock_sym
        self.window_size = window_size
        self.num_steps = num_steps
        self.test_percent = test_percent
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))

        # Merge into one array
        if close_price_only:
            self.raw_seq = np.array(raw_df['Close'].tolist())
        else:
            self.raw_seq = np.array([price for tup in raw_df[['Open', 'Close']].values for price in tup])

        #self.raw_seq = np.array(self.raw_seq)
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        # split seq into sliding windows
        num_windows = len(seq) // self.window_size
        seq = [np.array(seq[i * self.window_size: (i + 1) * self.window_size])
               for i in range(num_windows)]
        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]
            # [seq[0] / seq[0][0] - 1.0] for first window
            # [curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])] other windows, divided by last price of last window

        # split into groups of num_windows_input (the number of windows grouped in each input and each output)
        X = np.array([seq[i: i + self.num_windows_input] for i in range(len(seq) - self.num_windows_input)]) 
        y = np.array([seq[i + self.num_windows_input] for i in range(len(seq) - self.num_windows_input)])

        train_size = int(len(X) * (1.0 - self.test_percent))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
