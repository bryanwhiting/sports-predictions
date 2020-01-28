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
from config import config, save, load
import joblib
import pandas as pd
import scipy.stats as st
import xgboost
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

# from xgboost.sklearn import XGBRegressor
# pip install scikit-learn==0.18.2 - required for shap compatibility
import sklearn
import shap


def fit_model(
    target="result",
    n_sample: int = None,
    n_iter=100,
    retrain_mod_name=None,
    model_name="model_result_v02",
):
    """
    Debugging:
    target = 'result'

    retrain_mod_name: the name of the model object living in `/models`
    """
    dir_proc = config.get("dir", "proc")
    fp_mrd = os.path.join(dir_proc, "mrd_historical.csv")
    df = pd.read_csv(
        fp_mrd, parse_dates=["date", "datetime"], infer_datetime_format=True
    )
    df = df.sort_values(["datetime"])

    # TODO: one-hot encode team_abbr
    drop_cols = config.drop_cols + config.targets

    X_train = df[df.year <= 2017].drop(drop_cols, axis=1)
    X_eval = df[df.year == 2018].drop(drop_cols, axis=1)
    X_test = df[df.year == 2019].drop(drop_cols, axis=1)

    # Print out considered features
    print("Candidate features:", X_train.columns.tolist())

    # TODO: Bring in pg_home, pg_away to get spread and model the spread
    Y_train = df[df.year <= 2017][target]
    Y_eval = df[df.year == 2018][target]
    Y_test = df[df.year == 2019][target]

    if n_sample:
        X_train = X_train.sample(n_sample)
        Y_train = Y_train.sample(n_sample)

    # Scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    if target == "result":
        mod = xgboost.XGBClassifier(nthreads=1, verbosity=0)
        xgb_eval = ["logloss", "auc", "error"]
        gs_scoring = "accuracy"
    else:
        mod = xgboost.XGBRegressor(nthreads=1, verbosity=0)
        xgb_eval = "rmse"
        gs_scoring = "neg_root_mean_squared_error"

    # https://stackoverflow.com/a/57529837/2138773
    fit_params = {
        "early_stopping_rounds": 50,
        "eval_metric": xgb_eval,
        "eval_set": [(X_train, Y_train), (X_test, Y_test), (X_eval, Y_eval)],
    }

    if retrain_mod_name:
        # Train on an already-loded model
        xgb = load(dir="model", filename=retrain_mod_name)
        xgb.n_threads = -1
        xgb.fit(X_train, Y_train, **fit_params)
        final_mod = xgb

    else:
        # Do grid search:
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

        gs = RandomizedSearchCV(
            mod,
            paramGrid,
            verbose=5,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3),
            scoring=gs_scoring,
            random_state=1,
            n_jobs=-1,  # let the randomized search do the parallelization
        )

        gs.fit(X_train, Y_train, **fit_params)
        gs_results = pd.DataFrame(gs.cv_results_).sort_values(["rank_test_score"])

        save(gs_results, dir="model", filename=f"gs_{target}", main=False, date=True)
        print("gs_score:", gs.best_score_)
        # Return final model object
        final_mod = gs.best_estimator_

    # breakpoint()
    # Prediction sanity check
    # probs = pd.DataFrame(final_mod.predict_proba(X_test))
    # probs.columns = ['p' + str(c) for c in final_mod.classes_]
    # probs = probs[['p1']]
    # accuracy_score(Y_test, (probs > 0.4).astype('int'))
    # Sanity Check score:
    preds = pd.DataFrame(final_mod.predict(X_test))
    if target == "result":
        test_score = accuracy_score(Y_test, preds)
    else:
        test_score = mean_squared_error(Y_test, preds, squared=False)

    # Save out model:
    fn = model_name if model_name else f"model_{target}"
    save(final_mod, dir="model", filename=fn, main=True, date=True)
    return {"target": target, "test_score": test_score}


def plot_convergence(xgb):
    """Plot the convergence

    Todo: parameterize this so it's agnostic to the metric.
    Todo: save out to dir_ouc 
    """
    from matplotlib import pyplot
    import matplotlib.pyplot as plt

    # https://setscholars.net/wp-content/uploads/2019/02/visualise-XgBoost-model-with-learning-curves-in-Python.html
    plt.style.use("ggplot")
    results = final_mod.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)

    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
    ax.plot(x_axis, results["validation_2"]["logloss"], label="Eval")
    ax.legend()


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
    n_sample = None
    n_iter = 30
    retrain_mod_name = "model_result_v01"
    a = fit_model(
        target="result",
        n_sample=n_sample,
        n_iter=n_iter,
        retrain_mod_name=retrain_mod_name,
    )
    # c = fit_model(target="pg_spread", n_sample=n_sample, n_iter=n_iter)
    # b = fit_model(target="pg_score1", n_sample=n_sample, n_iter=n_iter)
    print(a)

    # View hyperparameter score:
    # load(dir='model', date='2020-01-21', filename='gs_pg_score1')
    # load(dir='model', date='2020-01-21', filename='gs_pg_score1')
