import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import numpy as np
import joblib
plt.style.use('fivethirtyeight')

def evaluate() -> None:
    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)
    
    _train = pd.read_csv(config__['data_split']['trainset_path'])
    _test = pd.read_csv(config__['data_split']['testset_path'])
   
    X_test = _test[_test.columns[0:-1]].values
    y_test = _test['PRICE'].values

    model_ = joblib.load(config__['train']['model_path'])
    preds_ = model_.predict(X_test)

    result_ = mean_squared_error(y_true=y_test, y_pred=preds_)
    result_ = np.sqrt(result_)
    print(f"Mean Squared Error {result_}")

    plt.figure(figsize = (10,6))
    sns.scatterplot(x = y_test, y = preds_)
    plt.savefig(config__['evaluate']['reports_dir'] + config__['evaluate']['scatterplot'])
    print('Reports saved to reports/figures')

    # print(X_test.shape, y_test.shape)

    if __name__ == '__main__':

        evaluate()