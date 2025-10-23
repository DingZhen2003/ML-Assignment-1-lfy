# solution.py
from typing import Dict, Any
from model import Model

# 全局加载一次模型（评测程序会多次调用 forward）
_model = None

def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model

class Solution:
    def forward(self, sample: Dict[str, Any]) -> Dict[str, float]:
        model = get_model()
        pred = model.predict(sample)
        return {'prediction': pred}