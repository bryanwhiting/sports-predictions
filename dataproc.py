"""

Steps:
* Load seasonal data
* Load aggregated season totals
* Add features within team. Lags, rolling sums.
    - lags
    - rolling sums
    - win pct
    - games in playoffs
    - opponent conference
    Season features
    - rank within season
    - conference strength: % of all wins
    - last season's total win pct
* Reshape data to be focused on home team only
    - This involves a self join. All opponent data needs to be renamed to opp_data.
    - 
* Save data out

Modeling:
* Build predictive model - xgboost: y = pr(home) wins
* Save out model

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
            pg_streak_int = lambda x: x.pg_streak_str.str.replace('L ', '-').str.replace('W ', '')
        )
)
df2.head(10)


# Time features
h_m_list = df.time.str.replace('p', '').str.split(':')
df3 = df2.assign(
    hr_int = h_m_list.apply(lambda x: x[0]).astype('int') + 12 - 1, # must be 0..23
    min_str = h_m_list.apply(lambda x: x[1]),
    min_int = lambda x: x.min_str.astype('int')/60,
    time_int = lambda x: x.hr_int + x.min_int,
    hr_str = lambda x: x.hr_int.astype('str'),
    ymdhms_str = lambda x: x.ymd_str + ' ' + x.hr_str + ':' + x.min_str + ':00',
    ymdhms = lambda x: pd.to_datetime(x.ymdhms_str)
    # TODO: add weekday
    # TODO: add weekend flag (sat/sun)
    # TODO: add holiday 
)

# Group by calculations - how do i??
# df3.group_by(['team_abbr', 'year']).agg()
# count number of games in the seasons
# compute fraction of season played

# Add lagged features, rolling features
shift_cols = ['wins']
grp_cols = ['team_abbr', 'year']
df3.groupby(['team_abbr', 'year']).shift(1)
# TODO?? how do I do this?
df3.groupby(['team_abbr', 'year'])['wins'].rolling(1).mean().reset_index()

# Produce final dataframe, which is just the home team.
# join on the boxscore_index FTW!!!
df_home = df3.query('home == 1')
# Drop anything from away that's duplicated and non-essential. Then rename to home_ and away_
df_away = df3.query('home == 0')
['boxscore_index',  'game', 
       'losses',  
        'result', 'streak', 'wins',
       'year', 'team_abbr', 'win_pct', 'home', 'hr_int', 'min_str', 'min_int',
       'time_int', 'hr_str', 'ymdhms_str', 'ymdhms']

# Drop anything from this dataset not used in the predictive model


