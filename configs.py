#configs.py

import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/captcha_to_text", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 40
        self.width = 150
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.train_epochs = 300
        self.train_workers = 20