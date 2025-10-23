# model.py
import numpy as np
import pickle

class Model:
    def __init__(self):
        with open('model.pkl', 'rb') as f:
            self.artifacts = pickle.load(f)
        self.w = self.artifacts['weights']
        self.b = self.artifacts['bias']
        self.cat_encoders = self.artifacts['cat_encoders']
        self.num_means = self.artifacts['num_means']
        self.num_stds = self.artifacts['num_stds']
        self.selected_idx = self.artifacts['selected_idx']
        self.cat_cols = self.artifacts['cat_cols']
        self.num_cols = self.artifacts['num_cols']

    def preprocess(self, sample):
        features = []

        # 类别特征：One-Hot
        for col in self.cat_cols:
            label = self.cat_encoders[col].get(sample[col], -1)
            for idx in self.cat_encoders[col].values():
                features.append(1 if label == idx else 0)

        # 数值特征
        num_vals = np.array([float(sample[col]) for col in self.num_cols])
        num_vals = (num_vals - self.num_means) / self.num_stds
        features.extend(num_vals.tolist())

        x = np.array(features)
        x = x[self.selected_idx]  # 特征选择
        return x

    def predict(self, sample):
        x = self.preprocess(sample)
        return float(np.dot(x, self.w) + self.b)