"""Model the data
2020-01-14: 45 min - got xgboost running for probabilities. Got shap values.

TODO:
* For a row, return the shap value in numeric form
*
"""

import os
from config import config
import pandas as pd
import xgboost

# from xgboost.sklearn import XGBRegressor
# pip install scikit-learn==0.18.2 - required for shap compatibility
import sklearn
import shap


dir_proc = config.get("dir", "proc")
fp_mrd = os.path.join(dir_proc, "2020-01-14", "mrd.csv")
df = pd.read_csv(fp_mrd)

# TODO: one-hot encode team_abbr
drop_cols = ["boxscore_index", "year", "ymdhms", "team_abbr", "opponent_abbr"]

X_train = df[df.year <= 2018].drop(drop_cols, axis=1).drop(["result"], axis=1)
X_test = df[df.year > 2018].drop(drop_cols, axis=1).drop(["result"], axis=1)
# TODO: Bring in pg_home, pg_away to get spread and model the spread
Y_train = df[df.year <= 2018]["result"]
Y_test = df[df.year > 2018]["result"]

# XGboost Dask-ml breaks this
# mod = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# from xgboost import XGBRegressor
# mod = XGBRegressor()
xgb_params = {"learning_rate": 0.01, "objective": "reg:squarederror"}
xgb_dat = xgboost.DMatrix(X_train.values, label=Y_train.values)
model = xgboost.train(xgb_params, xgb_dat, 100)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# View SHAP results
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[1000, :], X_train.iloc[1000, :])
# shap.force_plot(explainer.expected_value, shap_values, X_train)
shap.summary_plot(shap_values, X_train)

# with Flowk
#     Parameter('fp_mrd')
#     load_data()
