import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils')
from IO import *
from ML import *
from plot import *

seed = 42
test_size=0.25
n_trials=10
output_path = '../../output/'

def evaluate(y_true, y_pred):
    # AUC
    auc = metrics.roc_auc_score(y_true, y_pred)
    # # average_precision
    # average_precision = metrics.average_precision_score(y_true, y_pred)
    # # accuracy_score
    # accuracy = metrics.accuracy_score(y_true, np.rint(y_pred))
    return auc

def lgb_classifier(x_train, y_train):
    def objective(trial):
        train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=test_size)
        dtrain = lgb.Dataset(train_x, label=train_y)
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
        }
        clf = lgb.train(param, dtrain)
        pred_y = clf.predict(valid_x)
        score = evaluate(valid_y, pred_y)
        return score
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    dtrain = lgb.Dataset(x_train, label=y_train)
    clf = lgb.train(study.best_trial.params, dtrain)
    return clf

def lgb_plot(clf):
    for i in range(10):
        lgb.plot_tree(clf, tree_index=i)
        plt.savefig(output_path + f'tree_{i}.png')
        plt.clf()
        plt.close()

def main():
    initialize(output_path)
    # toyデータ作成
    import sklearn.datasets
    data = sklearn.datasets.load_iris()
    import pandas as pd
    df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
    df['y'] = data["target"]
    df = df[df['y']!=2]
    import random
    df['segment'] = [random.randint(0, 1) for _ in range(len(df))]
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop('y', axis=1), df['y'], test_size=test_size, random_state=seed
    )
    # toyデータ作成終了
    plot_histogram([df[df['y']==0]['sepal length (cm)'].values, df[df['y']==1]['sepal length (cm)'].values],'title',label_list=['a','b'])
    clf = lgb_classifier(x_train, y_train)
    y_pred = clf.predict(x_test)
    lgb_plot(clf)
    print(f'{metrics.roc_auc_score(y_test, y_pred)}, {metrics.average_precision_score(y_test, y_pred)}')
    plot_shap(clf, df.drop('y', axis=1))

if __name__ == '__main__':
    main()