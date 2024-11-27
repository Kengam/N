from sklearn.model_selection import train_test_split
import sklearn.metrics
import lightgbm as lgb
import optuna
seed = 42

def lgb_classifier(x_train, y_train):
    def objective(trial):
        train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.25)
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
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
        return accuracy
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    dtrain = lgb.Dataset(x_train, label=y_train)
    clf = lgb.train(study.best_trial.params, dtrain)
    return clf

def main():
    import sklearn.datasets
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)
    clf = lgb_classifier(x_train, y_train)
    y_pred = clf.predict(valid_x)
