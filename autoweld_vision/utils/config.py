import yaml
import os
from omegaconf import OmegaConf

class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        """Loads a config file using OmegaConf."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        config = OmegaConf.load(config_path)
        return config

    @staticmethod
    def get_base_config():
        base_path = "configs/base.yaml"
        return ConfigLoader.load_config(base_path)
