import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import joblib

def evaluate() -> None:
    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    feature_ = config__['featurize']['features_path']
    X_train = pd.read_csv(config__['data_split']['trainset_path'])
    X_test = pd.read_csv(config__['data_split']['testset_path'])
    y_train = X_train[config__['featurize']['target_column']]
    y_test = X_test[config__['featurize']['target_column']]

    scale_ = MinMaxScaler(feature_range=(0,1))
    scale_.fit(X_train)
    X_train = scale_.transform(X_train)
    X_test = scale_.transform(X_test) 

    model_ = joblib.load(config__['train']['model_path'])
    preds_ = model_.predict(X_test)

    plt.figure(figsize = (10,6))
    sns.regplot(x = preds_, y = y_test, line_kws = {'color': 'blue'}, scatter_kws = {'color': 'red'})

    plt.savefig(config__['evaluate']['reports_dir'] + ['regplot'])

    if __name__ == '__main__':

        evaluate()