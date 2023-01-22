import pandas as pd
import yaml


def data_load() -> None:
    with open('params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['data_load']['dataset_csv'])
    print("Data Successfully loaded")

if __name__ == '__main__':

    data_load()
