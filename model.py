"""Model the data
2020-01-14: 45 min - got xgboost running for probabilities. Got shap values.

TODO:
* For a row, return the shap value in numeric form
* There are only values back to 2000 in the MRD, should be 1995?

Notes:

XGboost Dask-ml breaks this:
# from xgboost import XGBRegressor
# mod = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)

"""

import os
from config import config, save
import joblib
import pandas as pd
import scipy.stats as st
import xgboost
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# from xgboost.sklearn import XGBRegressor
# pip install scikit-learn==0.18.2 - required for shap compatibility
import sklearn
import shap


def fit_model(target="result", n_sample: int = None):
    """
    Debugging:
    target = 'result'
    """
    dir_proc = config.get("dir", "proc")
    fp_mrd = os.path.join(dir_proc, "mrd_historical.csv")
    df = pd.read_csv(
        fp_mrd, parse_dates=["date", "datetime"], infer_datetime_format=True
    )
    df = df.sort_values(["datetime"])

    # TODO: one-hot encode team_abbr
    drop_cols = config.drop_cols

    X_train = df[df.year <= 2017].drop(drop_cols, axis=1).drop([target], axis=1)
    X_eval = df[df.year == 2018].drop(drop_cols, axis=1).drop([target], axis=1)
    X_eval = df[df.year == 2019].drop(drop_cols, axis=1).drop([target], axis=1)

    # TODO: Bring in pg_home, pg_away to get spread and model the spread
    Y_train = df[df.year <= 2017][target]
    Y_eval = df[df.year == 2018][target]
    Y_test = df[df.year == 2019][target]

    # http://danielhnyk.cz/how-to-use-xgboost-in-python/
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)

    paramGrid = {
        "n_estimators": [10000],
        "max_depth": st.randint(3, 40),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        "reg_alpha": from_zero_positive,
        "min_child_weight": [10, 25, 50, 75, 100, 150, 200],
    }

    if target == "result":
        mod = xgboost.XGBClassifier(nthreads=-1)
    else:
        mod = xgboost.XGBRegressor(nthreads=-1)

    # https://stackoverflow.com/a/57529837/2138773
    fit_params = {
        "early_stopping_rounds": 35,
        "eval_metric": "logloss",
        "eval_set": [[X_eval, Y_eval]],
    }

    gs = RandomizedSearchCV(
        mod,
        paramGrid,
        verbose=5,
        n_iter=5,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="accuracy",
        random_state=1,
    )

    gs.fit(X_train.sample(n=n_sample), Y_train.sample(n=n_sample), **fit_params)
    gs_results = pd.DataFrame(gs.cv_results_).sort_values(["rank_test_score"])

    final_mod = gs.best_estimator_

    # Prediction sanity check
    # probs = pd.DataFrame(final_mod.predict_proba(X_test))
    # probs.columns = ['p' + str(c) for c in final_mod.classes_]
    # probs = probs[['p1']]
    # Sanity Check score:
    # preds = pd.DataFrame(final_mod.predict(X_test))
    # accuracy_score(Y_test, preds)
    # accuracy_score(Y_test, (probs > 0.4).astype('int'))
    # gs.best_score_

    # Save out model:
    save(final_mod, dir="model", filename="model", main=True, date=False)


def SHAP():
    xgb_params = {"learning_rate": 0.01, "objective": "reg:squarederror"}
    xgb_dat = xgboost.DMatrix(X_train.values, label=Y_train.values)
    model = xgboost.train(xgb_params, xgb_dat, 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # View SHAP results
    shap.initjs()
    shap.force_plot(
        explainer.expected_value, shap_values[1000, :], X_train.iloc[1000, :]
    )
    # shap.force_plot(explainer.expected_value, shap_values, X_train)
    shap.summary_plot(shap_values, X_train)


if __name__ == "__main__":
    fit_model(target="result", n_sample=1000)
