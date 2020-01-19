"""

Steps:
* Load seasonal data
* Load aggregated season totals
* Add features within team. Lags, rolling sums.
    Rolling features:
    - 3-game sum of wins
    - 3-game streak (as opposed to overall streak)
    - 10-game record pct, sum of wins, etc.
    - average game time
    Lags:
    x lag a bunch of features
    x days since last game

    Game-specific features
    - win pct entering game
    - games in playoffs
    - opponent conference
    - home conference (one-hot-encoded)

    Season features
    - rank within season
    - conference strength: % of all wins
    - last season's total win pct
    - Total number of games in a season
        - Games remaining
        - % of games played so far

* Reshape data to be focused on home team only
    - This involves a self join. All opponent data needs to be renamed to opp_data.
    - 
* Save data out

Future:
* Cold-start problem. you can't use any lags for first game.
* If i use lag 5, I can't use first 5 games (unless I include back to prior season)

Modeling:
* Build predictive model - xgboost: y = pr(home) wins
* Save out model

Model will ignore:
* team effects (it only focuses on home team)
* But can capture matchups (home string vs away string?)

Production:
* Pull data for current season
* Engineer features for current season
"""

#%%

import logging
import os
import pandas as pd
from config import config

import prefect
from prefect import Flow, task, Parameter


def make_dir_proc():
    """Returns a folder for today's run"""
    dir_proc = os.path.join(config.get("dir", "proc"), config.get("date", "today"))
    os.makedirs(dir_proc, exist_ok=True)
    return dir_proc


@task
def read_in_data(fp_historical, nrows=None):
    """Specify manually which historical file you'll use"""
    df = pd.read_csv(fp_historical, nrows=nrows)

    # SANITY CHECKS
    assert (df.opponent_abbr == df.team_abbr).sum() == 0
    return df


@task
def basic_features(df):
    # breakpoint()
    df2 = (
        df.rename(
            {
                "datetime": "ymd_str",
                # pg = postgame. drop any of these from the model (they're not avaiable at start of game)
                # use these for feature engineering - lag
                "streak": "pg_streak_str",
                "points_allowed": "pg_pts_allowed",
                "points_scored": "pg_pts_scored",
            },
            axis=1,
        )
        .assign(
            win_pct=lambda x: x.wins / x.game,
            home=lambda x: (x.location == "Home").astype("int"),
            pg_streak_int=lambda x: x.pg_streak_str.str.replace("L ", "-").str.replace(
                "W ", ""
            ),
            result=lambda x: (x.result == "Win").astype("int"),
            playoffs=lambda x: x.playoffs.astype("int"),
        )
        .drop(["location"], axis=1)
    )
    return df2


from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


@task
def time_features(df):
    """Time features

    TODO: Consider adding all game-specific features here? 
    """
    logger = prefect.context.get("logger")

    # String splitting the time
    # h_m_list = df.time.str.replace('p', '').str.split(':')
    # df2 = df.assign(
    #     hr_str = h_m_list.apply(lambda x: x[0]),
    #     min_str = h_m_list.apply(lambda x: x[1]),
    #     hr_int = lambda x: x.hr_str.astype('int') + 12 - 1, # must be 0..23
    #     min_int = lambda x: x.min_str.astype('int')/60,
    #     time_int = lambda x: x.hr_int + x.min_int,
    #     ymdhms_str = lambda x: x.ymd_str + ' ' + x.hr_str + ':' + x.min_str + ':00',
    #     ymdhms = lambda x: pd.to_datetime(x.ymdhms_str)
    # ).drop(['min_int', 'min_str', 'hr_int', 'hr_str', 'ymdhms_str', 'date', 'ymd_str', 'time'], axis=1)

    # holidays
    cal = calendar()
    holidays = cal.holidays(start=df["ymd_str"].min(), end=df["ymd_str"].max())

    logger.info("Creating ymdhms, can take a while")
    df2 = df.assign(
        ymdhms_str=lambda x: x.ymd_str + " " + x.time,
        ymdhms=lambda x: pd.to_datetime(x.ymdhms_str),
    )
    logger.info("Creating other time features")
    df3 = df2.assign(
        # NaT is a missing time. This creates 19.5 for 7:30. fills NaT with 7pm
        time_int=lambda x: x.ymdhms.dt.strftime("%H")
        .str.replace("NaT", "19")
        .astype("int")
        + x.ymdhms.dt.strftime("%M").str.replace("NaT", "00").astype("int") / 60,
        holiday=lambda x: x.ymd_str.isin(holidays).astype("int"),
        # day of week. 5, 6 are sat, sun
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.dayofweek.html
        dayofweek=lambda x: x.ymdhms.dt.dayofweek,
        weekend=lambda x: x.dayofweek.isin([5, 6]),
    ).drop(["date", "time"], axis=1)
    return df3


@task
def lagged_features(df):
    """Add lagged features"""
    grp_cols = ["team_abbr", "year"]
    features_to_lag = [
        "pg_streak_int",
        "wins",
        "losses",
        "win_pct",
        "time_int",
        "ymdhms",
    ]
    df_lagged = df.groupby(grp_cols)[features_to_lag].shift(1)
    df_lagged.columns = ["lag1_" + c for c in df_lagged.columns]
    df2 = df.join(df_lagged)

    # Add advanced lagged features
    mask = (df2.ymdhms - df2.lag1_ymdhms).fillna(pd.Timedelta(seconds=0))
    # Converting to days:
    df2["lag1_days_since_game"] = mask.astype("int") / 1e9 / 60 / 60 / 24
    # Drop lag on times:
    df2 = df2.drop(
        [c for c in df2.columns if c.startswith("lag") and "ymdhms" in c], axis=1
    )
    return df2


@task
def rolling_features(df):
    # Add rolling features
    # TODO?? how do I do this?
    # df3.groupby(['team_abbr', 'year'])['wins'].rolling(1).mean().reset_index()
    return df


@task
def save_df(df, dir_proc, filename):
    """Save out the full and model-ready datasets to data/proc/YYYY-MM-DD/filename.csv
    
    TODO: Add this to a SQL database to save on memory 
    """
    df.to_csv(os.path.join(dir_proc, f"{filename}.csv"), index=False)


@task
def reshape_to_home(df):
    """Reshape the final dataset to be specifically for the home team

    Easily join on the boxscore_index FTW!!! 

    # Lets you see both home and away teams for a game
    df4[df4.boxscore_index == '201410310MIL']

    """
    # Drop anything from away that's duplicated and non-essential. Then rename to home_ and away_
    keys = ["boxscore_index", "year", "ymdhms", "team_abbr", "opponent_abbr", "result"]

    # Predictive features that are game-specific and known beforehand
    # (Don't have anything to do with the team)
    game_features = [
        "time_int",
        "playoffs",  # TODO: add day of week
    ]

    # Team-specific features. Only use features with lags bc they're in the past
    # TODO: add rolling features
    team_features = [c for c in df.columns if c.startswith("lag")]

    # Produce the home dataframe
    df_home = df.query("home == 1")[keys + game_features + team_features]
    # rename just the team features
    df_home.columns = (
        keys
        + game_features
        + ["home_" + c for c in df_home.columns if c in team_features]
    )

    # Produce the away dataframe
    df_away = df.query("home == 0")[["boxscore_index", "team_abbr"] + team_features]
    df_away.columns = ["boxscore_index", "away_team_abbr"] + [
        "away_" + c for c in df_away.columns if c in team_features
    ]

    # Merge the two dataframes on boxscore
    df_out = df_home.merge(df_away, on="boxscore_index")
    # SANITY CHECK: ensure the join is done properly. The away team_abbr should match the opponent of the home.
    # Since it does, we have a successful join!
    assert (df_out.opponent_abbr != df_out.away_team_abbr).sum() == 0, "Bad join"
    df_out = df_out.drop(["away_team_abbr"], axis=1)
    return df_out

@task
def fp_from_dir_raw(filename):
    return os.path.join(config.get('dir', 'raw'), filename)

# Create an output folder
dir_proc = make_dir_proc()

with Flow("Process data") as flow_proc:

    filename = Parameter("filename")
    nrow = Parameter("nrows")

    fp_data = fp_from_dir_raw(filename=filename)
    df = read_in_data(fp_data, nrows=nrow)
    df2 = basic_features(df)
    df3 = time_features(df2)
    df4 = lagged_features(df3)
    df5 = rolling_features(df4)  # PLACEHOLDER - doesn't do anything yet
    df6 = reshape_to_home(df5)
    # Save out
    save_full_df = save_df(df5, dir_proc=dir_proc, filename="full_data")
    save_mrd = save_df(df6, dir_proc=dir_proc, filename="mrd")


if __name__ == "__main__":
    # nrows = 10000
    nrows = None  # none runs whole dataset

    state = flow_proc.run(parameters={"nrows": nrows, "filename": "yr2020.csv"})  #
    fp_pdf = os.path.join(dir_proc, "flow.dot")
    flow_proc.visualize(flow_state=state, filename=fp_pdf)


# %%
