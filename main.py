import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree
import lightgbm
from lightgbm import LGBMClassifier
import optuna
import shap
import matplotlib.pyplot as plt
import sys
sys.path.append('../../utils')
from IO import *
from plot import *

import logging
logger = logging.getLogger()
logging.basicConfig(
    filename='log.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S'
)

seed = 42
test_size=0.25
n_trials=10
output_path = '../../output/'

def precision_score(conf_matrix):
    '''precision'''
    return np.diag(conf_matrix) / np.where(np.sum(conf_matrix,axis=0)==0,1,np.sum(conf_matrix,axis=0))

def recall_score(conf_matrix):
    '''recall'''
    return np.diag(conf_matrix) / np.where(np.sum(conf_matrix,axis=1)==0,1,np.sum(conf_matrix,axis=1))

def f1_score(conf_matrix):
    '''f1_score'''
    p = precision_score(conf_matrix)
    r = recall_score(conf_matrix)
    return (2*p*r) / np.where((p+r)==0,1,p+r)

def evaluate(y_true, y_pred):
    # AUC
    auc = metrics.roc_auc_score(y_true, y_pred)
    # # average_precision
    # average_precision = metrics.average_precision_score(y_true, y_pred)
    # # accuracy_score
    # accuracy = metrics.accuracy_score(y_true, np.rint(y_pred))
    # conf_matrix = confusion_matrix(y_true, y_pred) # 混同行列:(行:正解ラベル)*(列:予測ラベル)
    # score = precision_score(conf_matrix)
    # score = recall_score(conf_matrix)
    # score = f1_score(conf_matrix)
    return auc

def lgb_classifier(x_train, y_train):
    def objective(trial):
        train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=test_size, random_state=seed)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'seed': seed
        }
        clf = LGBMClassifier(**params)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(valid_x)
        score = evaluate(valid_y, pred_y)
        return score
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    clf = LGBMClassifier(**study.best_trial.params)
    clf.fit(x_train, y_train)
    return clf

def lgb_plot(clf):
    for i in range(10):
        lightgbm.plot_tree(clf, tree_index=i)
        plt.savefig(output_path + f'lgb_tree_{i}.png')
        plt.clf()
        plt.close()

def lgb_shap(clf, X, max_display=10):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.savefig(f'{args.output_path}/lgb_shap.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    for column in X.columns:
        shap.plots.scatter(shap_values[:, column], color=shap_values, show=False)
        plt.grid()
        plt.savefig(f'{args.output_path}/lgb_shap_{column}.png', bbox_inches='tight')
        plt.clf()
        plt.close()

def rf_classifier(x_train, y_train):
    def objective(trial):
        train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=test_size, random_state=seed)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'criterion': 'gini',
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'random_state': seed,
            'class_weight': 'balanced',
        }
        clf = RandomForestClassifier(**params)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(valid_x)
        score = evaluate(valid_y, pred_y)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    clf = RandomForestClassifier(**study.best_trial.params)
    clf.fit(x_train, y_train)
    return clf

def rf_plot(clf):
    for i in range(10):
        sklearn.tree.plot_tree(clf.estimators_[i])
        plt.savefig(output_path + f'rf_tree_{i}.png')
        plt.clf()
        plt.close()

def rf_shap(clf, X, max_display=10):
    0

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
    for classifier, plot, plot_shap in [
        (lgb_classifier, lgb_plot, lgb_shap),
        (rf_classifier, rf_plot, rf_shap)
        ]:
        clf = classifier(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(f'{metrics.roc_auc_score(y_test, y_pred)}, {metrics.average_precision_score(y_test, y_pred)}')
        logging.info(f'{metrics.roc_auc_score(y_test, y_pred)}, {metrics.average_precision_score(y_test, y_pred)}')
        plot(clf)
        plot_shap(clf, df.drop('y', axis=1))

if __name__ == '__main__':
    main()