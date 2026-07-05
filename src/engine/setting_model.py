import os
import sys
import importlib
from engine.config import Config as config

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss, L1Loss
from torch.optim import Adam, SGD, RMSprop, AdamW

class BaseModel():
    def __init__(self, hyperparams: config):
        self.hyperparams = hyperparams
        self.model_name = hyperparams.TrainingSetting.get('model')
        self.model_setting = hyperparams.ModelSetting
        self.device = hyperparams.TrainingSetting.get('device')
        self.target_params = hyperparams
        
    def _find_model_class(self):
        """
            Find target model in dir"model".
        """
        model = importlib.import_module(f"model.{self.model_name}")
        return getattr(model, self.model_name)

    def _set_model(self):
        params = self.transform_params()
        try:
            self.model = self._find_model_class()(**params)
        except:
            self.model = self._find_model_class()(params)
        return self.model
        
    def transform_params(self):
        return self.target_params
    
    def optimizer(self):
        opt_name = self.hyperparams.TrainingSetting.get('optimizer')
        lr = self.hyperparams.TrainingSetting.get('learning_rate')
        weight_decay = self.hyperparams.TrainingSetting.get('weight_decay', 0)
        if opt_name == "Adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_name == "SGD":
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_name == "RMSprop":
            return RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_name == "AdamW":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_name in globals():
            return globals()[opt_name](self.model.parameters(), lr=lr)
        raise RuntimeError(f"Optimizer '{opt_name}' not found.")
    
    def get_model(self):
        model = self._set_model()
        optimizer = self.optimizer()
        if self.device == "cuda" or self.device == "gpu" or self.device == "cuda:0":
            model = model.cuda()
        return (model, optimizer)

        
def choose_model(cfg):
    model_name = cfg.TrainingSetting.get('model')
    model = BaseModel(cfg)
    model.model_name = model_name
    return model.get_model()

