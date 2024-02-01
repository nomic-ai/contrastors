import yaml
from contrastors.config import Config


def read_config(path):
    # read yaml and return contents
    with open(path, 'r') as file:
        try:
            return Config(**yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)
