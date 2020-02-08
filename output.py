"""
produce markdown

1. Read in 2020 data
2.

# Todo:
- Make graph of accuracy over time
    - Add a feature which is season_week
    - Group by season_week, calculate accuracy for all model types.

"""
import os
import shutil
from config import config, save, load
import pandas as pd
import numpy as np
from prefect import task, Flow
from datetime import date
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from xgboost import plot_importance


# @task
def plot_imp():
    xgb = load(dir="model", filename="model_result")
    plot_importance(xgb, importance_type="gain")
    pyplot.show()


def score_df(df):
    """ Load in the MRD and score"""
    X = df.drop(config.drop_cols + config.targets, axis=1)

    # Binary predictions. Add model objects here
    for model in ["result_v01", "result_v02"]:
        xgb = load(dir="model", filename=f"model_{model}")
        # Extract the features used in the model
        feats = xgb.get_booster().feature_names
        X_mod = X[feats]
        # Predict only on the model's features
        df[f"{model}_prob1"] = pd.DataFrame(xgb.predict_proba(X_mod))[1]
        # TODO: Optimize threshold on 2019 data to maximize accuracy.
        # TODO: consider other metrics beyond accuracy to measure.
        df[f"pred_{model}_prob1"] = xgb.predict(X_mod)
        df[f"acc_pred_{model}_prob1"] = xgb.predict(X_mod) == df.result

    # Spread model
    # xgb_spread = load(dir='model', filename='model_pg_spread')
    # df['v01_spread'] = xgb_spread.predict(X)
    return df


def df_with_538():
    """Create a combined dataset
    Pulls from mrd_yr2020 and 538, checks they have the same scores
    and then merges them together

    df5 = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv')
    df5 = df5[['date', 'team1', 'team2', 'elo_prob1', 'carm-elo_prob1', 'raptor_prob1', 'score1', 'score2']]
    """
    # My MRD
    mrd2020 = config.get("dir", "mrd2020")
    df = pd.read_csv(mrd2020)

    # 538 dataset:
    df5 = pd.read_csv(
        "https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv"
    )
    df5 = df5[
        [
            "date",
            "team1",
            "team2",
            "elo_prob1",
            "carm-elo_prob1",
            "raptor_prob1",
            "score1",
            "score2",
        ]
    ]
    # Combined
    df = df.merge(df5, on=["date", "team1", "team2"])

    if len(df[df["pg_score1"] != df["score1"]]) > 0:
        print("WARNING score1 differs. consider re-pulling your data")

    if len(df[df["pg_score2"] != df["score2"]]) > 0:
        print("WARNING score2 differs. consider re-pulling your data")

    # Remove duplicated 538 columns
    df = df.drop(["score1", "score2"], axis=1)

    # Calculate season_week
    df["season_week"] = (pd.to_datetime(df.date).dt.date - date(2019, 10, 20)).apply(
        lambda x: x.days // 7
    ) + 1
    return df


def breakdown_accuracy(df, result, proba, pred, name):
    """Take a model score and break it down.
    pred is a 0/1 column, where proba is the probabilty score for home team/team1"""

    # Filter just to today's date
    df = df[df.date < config.get("date", "today")].copy()
    acc = {}
    acc["all"] = [accuracy_score(y_true=df[result], y_pred=df[pred])]
    for i in np.arange(0.0, 1.0, 0.1):
        min = round(i, 1)
        max = round(min + 0.1, 1)
        df_filt = df[(min < df[proba]) & (df[proba] <= max)]
        # Returns calibration (should be between bounds), which is 1-acc when prob < 0.5
        _acc = df_filt.result.mean()
        # _acc = accuracy_score(y_true = df_filt[result], y_pred=df_filt[pred])
        acc[f"{str(min)}:{str(max)}"] = [_acc]

    df_acc = pd.DataFrame(acc)
    df_acc.index = [name]
    return df_acc


# @task
def benchmark_model_accuracy(df):
    """Extract the win percentage

    Compare accuracy with result.mean
    x = df[(df.elo_prob1 > 0.2) & (df.elo_prob1 < 0.3)]
    # Accuracy:
    print(accuracy_score(x.result, x.elo_prob1 > .5))
    print(x.result == (x.elo_prob1 > 0.5).astype('int'))
    # Calibration (should be between 0.2 and 0.3 if calibrated)
    print(x.result.mean())
    """

    # Prediction = Home team win pct > 0.5
    df["pred_home_winpct"] = (df["team1_lag1_win_pct"] > 0.5).astype("int")
    df["acc_pred_home_winpct"] = (df["result"] == df["pred_home_winpct"]).astype("int")

    acc_home_win_pct = breakdown_accuracy(
        df,
        result="result",
        proba="team1_lag1_win_pct",
        pred="pred_home_winpct",
        name="home_winpct",
    )

    # Don't use breakdown_accuracy() function for these simple aggregates
    df_tmp = df[df.date < config.get("date", "today")].copy()
    # home team win pct
    acc_home = df_tmp.result.mean()
    # home win pct > away win pct.
    pred = (df_tmp["team1_lag1_win_pct"] > df_tmp["team2_lag1_win_pct"]).astype("int")
    acc_home_vs_away = accuracy_score(y_true=df_tmp.result, y_pred=pred)

    # Combine all accuracies
    df_acc = acc_home_win_pct.append(
        pd.DataFrame(
            {"all": [acc_home, acc_home_vs_away]},
            index=["home_overall", "home_vs_away_winpct"],
        ),
        sort=False,
    )

    # Add Bryan's predictions
    for ver in ["v01", "v02"]:
        acc_ver = breakdown_accuracy(
            df,
            result="result",
            proba=f"result_{ver}_prob1",
            pred=f"pred_result_{ver}_prob1",
            name=ver,
        )
        df_acc = df_acc.append(acc_ver)

    # Does prediction improve after filtering out first month?
    # df = df[df.ymdhms > '2019-12-01']
    # pred = (df['home_lag1_win_pct'] > .5).astype('int')
    # accuracy_score(y_true=df.result, y_pred=pred)

    # Predicting using the away team win pct
    # pred = 1-(df['away_lag1_win_pct'] > .5).astype('int')
    # accuracy_score(y_true=df.result, y_pred=pred)

    # Five-thirty-eight models
    # Compare Nate Silver's results
    models = ["elo", "carm-elo", "raptor"]
    for model in models:
        df[f"pred_{model}_prob1"] = (df[f"{model}_prob1"] > 0.5).astype("int")
        acc = breakdown_accuracy(
            df,
            result="result",
            proba=f"{model}_prob1",
            pred=f"pred_{model}_prob1",
            name=model,
        )
        df_acc = df_acc.append(acc)

        # Save out binary accuracy for groupby-accuracy by season_week later
        df[f"acc_pred_{model}_prob1"] = (
            df[f"pred_{model}_prob1"] == df["result"]
        ).astype("int")

    # reset index because rownames contain the accuracy breakdowns
    df_acc = df_acc.transpose().fillna("").reset_index()
    return df, df_acc


# @task
def week_accuracies(df):

    # Filter just to today's copy
    df = df[df.date < config.get("date", "today")]
    pred_cols = [c for c in df.columns if c.startswith("acc_pred_")]
    # Within-week accuracies
    df_within_week = df.groupby(["season_week"])[pred_cols].mean().reset_index()

    # up-to-week accuracy
    max_weeks = df.season_week.max()
    df_upto_week = pd.DataFrame()
    for i in range(0, max_weeks + 1):
        df_filt = df[df.season_week <= i]
        df_week = pd.DataFrame(df_filt[pred_cols].mean()).transpose()
        df_week["season_week"] = i
        df_upto_week = df_upto_week.append(df_week)

    df_upto_week = df_upto_week[["season_week"] + pred_cols]

    # Since week accuracy # SAME AS CODE ABOVE
    max_weeks = df.season_week.max()
    df_since_week = pd.DataFrame()
    for i in range(0, max_weeks + 1):
        df_filt = df[df.season_week >= i]  # ONLY CHANGE
        df_week = pd.DataFrame(df_filt[pred_cols].mean()).transpose()
        df_week["season_week"] = i
        df_since_week = df_since_week.append(df_week)

    df_since_week = df_since_week[["season_week"] + pred_cols]

    save(
        df_within_week,
        dir="output",
        filename="acc_within_week",
        ext=".csv",
        main=True,
        date=False,
    )
    save(
        df_upto_week,
        dir="output",
        filename="acc_upto_week",
        ext=".csv",
        main=True,
        date=False,
    )
    save(
        df_since_week,
        dir="output",
        filename="acc_since_week",
        ext=".csv",
        main=True,
        date=False,
    )

    # return (df_within_week, df_upto_week)


def save_scores(df):
    # attributes to save
    keep_cols = [
        "datetime",
        "season_week",
        "team1",
        "team2",
        "result",
        "pg_score1",
        "pg_score2",
        "elo_prob1",
        "carm-elo_prob1",
        "raptor_prob1",
    ]
    # my models
    keep_cols += [
        c for c in df.columns if c.startswith("result_") and c.endswith("prob1")
    ]
    #
    df_out = df[keep_cols].sort_values(["datetime", "team1"])
    df_out.loc[df_out.datetime >= config.get("date", "today"), "result"] = ""
    df_out = df_out.fillna("")
    # View today's games
    # df_out[df_out.datetime >= config.get('date', 'today')].head(20)
    save(df_out, dir="output", filename="all_scores", ext=".csv", main=True, date=False)


def get_538_standings(freq=True):
    """Pull down the rankings from 538"""
    df_stand = pd.read_html(
        "https://projects.fivethirtyeight.com/2020-nba-predictions/"
    )[0]

    df_stand.columns = df_stand.columns.get_level_values(0)
    df_stand.columns = [
        "elo",
        "x",
        "x",
        "team",
        "conf",
        "rating_season",
        "record_proj_538_raptor",
        "proj_pt_diff",
        "prob_playoff",
        "rating_playoffs",
        "prob_finals",
        "prob_win_final",
        "x",
        "x",
    ]
    df_stand = df_stand[[c for c in df_stand.columns if c != "x"]]

    # Extract just the team name
    df_stand["name"] = df_stand["team"].str.extract("(^.*[a-z]+)[0-9]+-")
    df_stand["538_curr_record"] = df_stand["team"].str.extract("^.*[a-z]+([0-9]+-.*)")
    # Add the win_pct
    if freq:
        # CURRENT RECORD
        win = df_stand["team"].str.extract("^.*[a-z]+([0-9]+)-.*").astype("int")
        loss = df_stand["team"].str.extract("^.*[a-z]+[0-9]+-([0-9]+)").astype("int")
        pct = round(win / (win + loss) * 100).astype("int").astype("str")
        df_stand["pct"] = " (" + pct + "%)"
        df_stand["538_curr_record"] = df_stand["538_curr_record"] + df_stand["pct"]

        # PROJ 538 RECORD (copied )
        win = (
            df_stand["record_proj_538_raptor"].str.extract("^([0-9]+)-.*").astype("int")
        )
        loss = (
            df_stand["record_proj_538_raptor"]
            .str.extract("^[0-9]+-([0-9]+)")
            .astype("int")
        )
        pct = round(win / (win + loss) * 100).astype("int").astype("str")
        df_stand["pct"] = " (" + pct + "%)"
        df_stand["record_proj_538_raptor"] = (
            df_stand["record_proj_538_raptor"] + df_stand["pct"]
        )

    # There is a lot in this table: elo, conf, rating_season
    df_stand = df_stand[["name", "record_proj_538_raptor", "538_curr_record"]]

    # Get abbrv for merging
    df_names = pd.read_csv(config.get("dir", "names"))[["team", "name"]]
    df_stand = df_stand.merge(df_names, on="name")
    return df_stand


def create_record(df, pred_col, freq=True):
    """Takes a prediction column and calculates projected record

    TODO: add datetime so you can show a graph of which game they'll win
    """
    x = df[df[pred_col].notnull()][["team1", "team2", pred_col]].copy()
    # rename columns
    x.columns = ["team1", "team2", "pred1"]
    x["pred1"] = x["pred1"].astype("int")
    x["pred2"] = (x["pred1"] == 0).astype("int")
    # split data into team1 and team2 to have wins/losses long
    team1 = x[["team1", "pred1"]].rename({"team1": "team", "pred1": "pred"}, axis=1)
    team2 = x[["team2", "pred2"]].rename({"team2": "team", "pred2": "pred"}, axis=1)
    teams = team1.append(team2)
    # Calculate record
    teams = teams.groupby(["team"]).agg({"pred": ["sum", "count"]}).reset_index()
    # Calculate win/loss
    teams.columns = teams.columns.get_level_values(0)
    teams.columns = ["team", "wins", "n"]
    teams = teams.assign(
        losses=lambda x: x.n - x.wins,
        freq=lambda x: round(x.wins / (x.n) * 100).astype("int").astype("str"),
        record=lambda x: x.wins.astype("str") + "-" + x.losses.astype("str"),
    )
    if freq:
        teams.record = teams.record + " (" + teams.freq + "%)"
    teams = teams[["team", "record"]].rename({"record": "record_" + pred_col}, axis=1)
    return teams


def all_records(df_scored2, freq):
    # columns for prediction
    pred_cols = [c for c in df_scored2.columns if c.startswith("pred_")]
    cols = ["team1", "team2", "result", "datetime"] + pred_cols
    df_preds = df_scored2[cols].copy()
    df_preds.loc[df_preds.datetime > config.get("date", "today"), "result"] = None

    # calculate record
    df_records = create_record(df_preds, pred_col="result", freq=freq)
    # conditionally replace the prediction columns
    for c in pred_cols:
        df_preds[c] = np.where(df_preds.result.isna(), df_preds[c], df_preds.result)
        df_rec = create_record(df_preds, pred_col=c, freq=freq)
        df_records = df_records.merge(df_rec, on="team")

    # Join 538
    # Get 538 standings
    df_538_stand = get_538_standings(freq=freq)
    df_records = df_records.merge(df_538_stand, on="team")

    # Sanity check on the record (to make sure calculated correctly with 538)
    equals = len(
        df_records[df_records["record_result"] != df_records["538_curr_record"]]
    )
    if equals != 0:
        print(
            "WARNING: Your RECORD calculation does not equals 538's. Using 538 instead."
        )
        df_records["record_result"] = df_records["538_curr_record"]

    order_cols = ["team", "name"]
    df_records = df_records[
        order_cols + [c for c in df_records.columns if c not in order_cols]
    ]
    return df_records


@task
def main():
    df = df_with_538()
    df_scored = score_df(df)
    # Each of these functions is updating df
    df_scored2, df_acc = benchmark_model_accuracy(df_scored)
    # pandas groupby means
    _ = week_accuracies(df_scored2)

    df_records = all_records(df_scored2, freq=True)

    # Save out
    save_scores(df_scored)

    save(
        df_acc, dir="output", filename="acc_overall", ext=".csv", main=True, date=False
    )
    save(
        df_records, dir="output", filename="records", ext=".csv", main=True, date=False
    )


@task
def copy_data():
    # Ignore subdirectories
    src_dir = config.get("dir", "output")
    out_dir = os.path.join(config.get("dir", "site"), "data")
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
    for f in files:
        src_path = os.path.join(src_dir, f)
        out_path = os.path.join(out_dir, f)
        print("Copying from:", src_path, " to:", out_path)
        shutil.copy(src_path, out_path)


with Flow("Build Output") as flow_output:
    # NOTE TO SELF:
    # I didn't use prefect for individual tasks becasue I saw
    # some reallys trange behavior where the accuracies weren't saving out
    # That, and I like putting breakpoints in the main to see
    # how data are passed from one task to another
    m = main()
    c = copy_data(upstream_tasks=[m])


if __name__ == "__main__":
    state = flow_output.run()
    # Debug:
    # state.result[df_weeks].result
