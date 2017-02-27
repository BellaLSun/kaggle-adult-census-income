# encoding: UTF-8

import numpy as np
import pandas as pd
from prediction.loader import read_income_data
from prediction.algo import get_predictions
from prediction.algo import get_regressor
from prediction.algo import get_metrics


def get_accuracy_per_n_estimator(algo, n_estimators, cv=30):
    df = read_income_data(True)
    target = df.pop('income')
    features = df
    results = []
    for i in n_estimators:
        clf_rf = get_regressor(algo, {"n_estimators": i, "n_jobs": 2})
        p_rf = get_predictions(clf_rf, features, target, cv)
        _, _, _, _, auc_rf = get_metrics(target, p_rf)
        results.append(auc_rf)

    return results
