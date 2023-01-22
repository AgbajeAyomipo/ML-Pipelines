import pandas as pd
import yaml
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler


def data_split() -> None:
    with open('../params.yaml') as config_:
        config__ = yaml.safe_load(config_)

    df_ = pd.read_csv(config__['featurize']['features_path'])

    X = df_.copy()

    X_train, X_test = train_test_split(X, test_size = 0.25, random_state = config__['base']['random_state'])

    X_train_df = pd.DataFrame(data = X_train, columns = df_.drop('PRICE', axis = 1).columns)
    X_test_df = pd.DataFrame(data = X_test, columns = df_.drop('PRICE', axis = 1).columns)
    X_train_df.to_csv(config__['data_split']['trainset_path'])
    X_test_df.to_csv(config__['data_split']['testset_path'])

    process_path = 'data/processed'
    print("Data has been splitted and successfully saved to process_path")

    if __name__ == '__main__':

        data_split()