import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pandas as pd
import yaml
import joblib
import os


os.chdir('C:/Users/Ayo Agbaje/Desktop/ML-Pipelines/notebooks')
def train_() -> None:
    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    _train = pd.read_csv(config__['data_split']['trainset_path'])
    _test = pd.read_csv(config__['data_split']['testset_path'])
   
    X_train = _train[_train.columns[0:-1]].values
    X_test = _test[_test.columns[0:-1]].values
    y_train = _train['PRICE'].values

    rfr = RandomForestRegressor(
        n_estimators=config__['train']['n_estimators'],
        n_jobs=config__['train']['n_jobs'],
        max_features=config__['train']['max_leaf_nodes']
    )
    rfr.fit(X_train, y_train)
    joblib.dump(rfr, config__['train']['model_path'])

    print("Model Successfully trained and saved")

if __name__ == '__main__':

    train_()