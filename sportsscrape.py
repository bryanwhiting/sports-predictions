import os
import time
import itertools

import pandas as pd
import prefect
from prefect import task, Flow
from sportsreference.nba.teams import Teams
from sportsreference.nba.schedule import Schedule

from config import config

@task
def get_list_of_team_abbrs():
    try:
        teams = Teams()
        teams_df = teams.dataframes
        abbr = list(teams_df.abbreviation.values)
    except:
        abbr = ['MIL', 'LAC', 'DAL', 'HOU', 'NOP', 'MEM', 'POR', 'WAS', 'PHO',
       'TOR', 'CHO', 'LAL', 'PHI', 'IND', 'MIA', 'DET', 'SAS', 'ATL',
       'GSW', 'DEN', 'SAC', 'UTA', 'OKC', 'MIN', 'CHI', 'NYK', 'BRK',
       'ORL', 'BOS', 'CLE']
    return sorted(abbr)

@task
def create_year_abbr_combo(abbrs, from_year, thru_year):
    years = list(range(from_year, thru_year + 1))
    years_abbr = list(itertools.product(years, abbrs))
    return years_abbr

@task
def return_schedule_dataframe(year_abbr: tuple):
    year = year_abbr[0]
    abbr = year_abbr[1]
    sched = Schedule(abbr, year=year)
    df_sched = sched.dataframe
    df_sched['team_abbr'] = abbr
    df_sched['year'] = year
    return df_sched

@task
def produce_single_df():
    # TODO:
    dir_out = os.path.join(config.get(dir_raw), config.get('date', 'today'))
    os.makedirs(dir_out, exist_ok=True)
    df.to_csv('')

with Flow('Raw Pull') as flow_hist:
    abbrs = get_list_of_team_abbrs()
    year_abbrs = create_year_abbr_combo(abbrs, from_year=2015, thru_year=2019)
    dfs = return_schedule_dataframe.map(year_abbrs)
    # TODO: Write these to a sqllite database
    # TODO: Reduce into a single data frame
    # Combine them into a single data set

flow_hist.visualize()
# flow_hist.run()

df = pd.DataFrame()
cnt = 1
for abbr, year in year_abbr:
    print(abbr, year)
    sched = Schedule(abbr, year=year)
    print('getting dataframe')
    df_sched = sched.dataframe
    df_sched['team_abbr'] = abbr
    df_sched['year'] = year
    df = df.append(df_sched)
    print('writing out')
    df.to_csv('~/Desktop/historical2_dataframe.csv', index=False)
    cnt += 1

    # if cnt > 5:
    #     break
    # time.sleep(1)

