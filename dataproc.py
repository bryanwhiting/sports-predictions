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

import os
import pandas as pd
from config import config

fp_raw = os.path.join(config.get('dir', 'raw'), '20200109/historical.csv')
df = pd.read_csv(fp_raw)

# SANITY CHECKS
assert (df.opponent_abbr == df.team_abbr).sum() == 0
assert df.time.str.contains('a').sum() == 0

# basic features
df2 = (df.
        rename({
            'datetime': 'ymd_str',
            #pg = postgame. drop any of these from the model (they're not avaiable at start of game)
            # use these for feature engineering - lag
            'streak': 'pg_streak_str', 
            'points_allowed': 'pg_pts_allowed',
            'points_scored': 'pg_pts_scored',
        }, axis=1).
        assign(
            win_pct = lambda x: x.wins/x.game,
            home = lambda x: (x.location == 'Home').astype('int'),
            pg_streak_int = lambda x: x.pg_streak_str.str.replace('L ', '-').str.replace('W ', ''),
            result = lambda x: (x.result == 'Win').astype('int'),
        ).drop(['location'], axis=1)
)
df2.head(10)


# Time features
h_m_list = df.time.str.replace('p', '').str.split(':')
df3 = df2.assign(
    hr_str = h_m_list.apply(lambda x: x[0]),
    min_str = h_m_list.apply(lambda x: x[1]),
    hr_int = lambda x: x.hr_str.astype('int') + 12 - 1, # must be 0..23
    min_int = lambda x: x.min_str.astype('int')/60,
    time_int = lambda x: x.hr_int + x.min_int,
    ymdhms_str = lambda x: x.ymd_str + ' ' + x.hr_str + ':' + x.min_str + ':00',
    ymdhms = lambda x: pd.to_datetime(x.ymdhms_str)
    # TODO: add weekday
    # TODO: add weekend flag (sat/sun)
    # TODO: add holiday 
).drop(['min_int', 'min_str', 'hr_int', 'hr_str', 'ymdhms_str', 'date', 'ymd_str', 'time'], axis=1)

# Group by calculations - how do i??

# Add lagged features
grp_cols = ['team_abbr', 'year']
features_to_lag = ['pg_streak_int', 'wins', 'losses', 'win_pct', 'time_int', 'ymdhms']
df_lagged = df3.groupby(grp_cols)[features_to_lag].shift(1)
df_lagged.columns = ['lag1_' + c for c in df_lagged.columns]
df4 = df3.join(df_lagged)

# Add advanced lagged features
# Converting to days:
df4['lag1_days_since_game'] = (df4.ymdhms - df4.lag1_ymdhms).fillna(pd.Timedelta(seconds=0)).astype('int')/1e9/60/60/24

# Add rolling features
# TODO?? how do I do this?
df3.groupby(['team_abbr', 'year'])['wins'].rolling(1).mean().reset_index()


# TODO: Save out this final, full dataset for by-team analysis
dir_proc = os.path.join(config.get('dir', 'proc'), config.get('date', 'today'))
os.makedirs(dir_proc, exist_ok=True)
# df4.to_csv(os.path.join(dir_proc, 'full_team.csv'), index=False)

# Produce final dataframe, which is just the home team.
# join on the boxscore_index FTW!!!
df4[df4.boxscore_index == '201410310MIL']

#
# Drop anything from away that's duplicated and non-essential. Then rename to home_ and away_

keys = ['boxscore_index', 'year', 'ymdhms', 'team_abbr', 'opponent_abbr', 'result']
df_home[keys]

game_features = [
    'time_int',
    'playoffs', # TODO: add day of week 
] 
team_features = [c for c in df4.columns if c.startswith('lag')]
# team features to add

df_home = df4.query('home == 1')[keys + game_features + team_features]
# rename just the team features
df_home.columns = keys + game_features + ['home_' + c for c in df_home.columns if c in team_features]

df_away = df4.query('home == 0')[['boxscore_index', 'team_abbr'] + team_features]
df_away.columns = ['boxscore_index', 'away_team_abbr'] + ['away_' + c for c in df_away.columns if c in team_features]

df_out = df_home.merge(df_away, on='boxscore_index')
# SANITY CHECK: ensure the join is done properly
assert (df_out.opponent_abbr != df_out.away_team_abbr).sum() == 0
df_out = df_out.drop(['away_team_abbr'], axis=1)

# DONE!!!

# Q; three types of features: game-specific features only the home team should have (time of day, day of week)
# lagged features should all be demarked with 'home_'
['boxscore_index', 'game', 
       'losses',  
        'streak', 'wins',
       'year', 'team_abbr', 'win_pct', 'home', 'hr_int', 'min_str', 'min_int',
       'time_int', 'hr_str', 'ymdhms_str', 'ymdhms']

# Drop anything from this dataset not used in the predictive model



