import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pandas as pd
import yaml
import joblib


def train_() -> None:
    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    X_train = pd.read_csv(config__['data_split']['trainset_path']).drop('PRICE', axis = 1)
    X_test = pd.read_csv(config__['data_split']['testset_path']).drop('PRICE', axis = 1)
    y_train = X_train[config__['featurize']['target_column']]
    y_test = X_test[config__['featurize']['target_column']]

    scale_ = MinMaxScaler(feature_range=(0,1))
    scale_.fit(X_train)
    X_train = scale_.transform(X_train)
    X_test = scale_.transform(X_test) 

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