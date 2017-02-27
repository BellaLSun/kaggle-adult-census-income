# encoding: UTF-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

matplotlib.style.use('ggplot')


def get_regressor(regressor, params={}):
    if regressor == "random_forest":
        args = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 0,
            "oob_score": True,
            "n_jobs": -1
        }
        args.update(params)
        return RandomForestRegressor(**args)
    if regressor == "gradient_boosting":
        args = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 0,
            "loss": 'ls',
            "subsample": 1.0
        }
        args.update(params)
        return GradientBoostingRegressor(**args)
    if regressor == "linear":
        return LinearRegression()


def get_confusion_matrix(target, predictions, threshold, normalize=True):
    cm = metrics.confusion_matrix(target, predictions >= threshold)
    if normalize:
        # Last minute not working
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    return pd.DataFrame(cm).T


def get_feature_importance(regressor, features, target):
    regressor.fit(features, target)
    return pd.DataFrame(zip(
        features.columns,
        regressor.feature_importances_)).sort_values(1, ascending=False)


def get_predictions(regressor, features, target, cv=30):
    return cross_val_predict(regressor, features, target, n_jobs=-1, cv=cv)


def get_metrics(target, predictions):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    precision_score = metrics.average_precision_score(target, predictions)
    return (
        false_positive_rate,
        true_positive_rate,
        thresholds,
        precision_score,
        roc_auc
    )
