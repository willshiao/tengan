'''Defines the ModelZoo class, which allows for easy switching of models'''

from ggan.rescal import RescalGenerator
from .tgan import NewLayeredMultiviewDiscriminator
from .new_tgan import NewCPTensorGenerator

class ModelZoo:
    def __init__(self):
        self.models = {
            'NewCPTensorGenerator': NewCPTensorGenerator,
            'NewLayeredMultiviewDiscriminator': NewLayeredMultiviewDiscriminator,
            'RescalGenerator': RescalGenerator
        }

    def get_model(self, model_name):
        '''Given a model name, returns the model class'''
        if model_name not in self.models:
            raise Exception(
                f'Unknown model specified. Valid options are: {self.models.keys()}')
        return self.models[model_name]

    def has_model(self, model_name):
        '''Given a model name, return whether or not it exists'''
        return model_name in self.models

