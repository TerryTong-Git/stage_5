'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import torch

class GNN_Setting(setting):
    
    def load_run_save_evaluate(self, *argv):
        # run MethodModule


        learned_result = self.method.run(*argv)

        return
      