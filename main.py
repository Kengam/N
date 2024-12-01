import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils')
from IO import *

seed = 42
test_size=0.25
n_trials=10
output_path = '../../output/'

def evaluate_binaryClassification(y_true, y_pred):
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
        score = evaluate_binaryClassification(valid_y, pred_y)
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
    import sklearn.datasets
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=seed)
    clf = lgb_classifier(x_train, y_train)
    y_pred = clf.predict(x_test)
    lgb_plot(clf)
    print(f'{metrics.roc_auc_score(y_test, y_pred)}, {metrics.average_precision_score(y_test, y_pred)}')

if __name__ == '__main__':
    main()