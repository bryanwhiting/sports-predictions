from sportsreference.nba.teams import Teams
from sportsreference.nba.schedule import Schedule

# For a team:
gsw = Schedule("GSW", year=2018)
x = gsw.dataframe


teams = Teams()
teams.dataframes


teams = Teams()
for team in teams:
    print(team.name)

for team in teams:
    schedule = team.schedule  # Returns a Schedule instance for each team
    # Returns a Pandas DataFrame of all metrics for all game Boxscores for
    # a season.
    df = team.schedule.dataframe_extended


# Read in the scraped data
import pandas as pd

df = pd.read_csv("~/Desktop/historical_dataframe.csv")

# 1) I can tack on the abbreviations (which I have for every year up through Chicago)
# - Then assert that abbr != opponen_abbr
# Get the team annual schedules (results) - lag the year and then join on year (to get prior year's) results

# Feature engineering
# Win % (win / game)
# Streak


# HOTFIX THE DATA
abbr = [
    "MIL",
    "LAC",
    "DAL",
    "HOU",
    "NOP",
    "MEM",
    "POR",
    "WAS",
    "PHO",
    "TOR",
    "CHO",
    "LAL",
    "PHI",
    "IND",
    "MIA",
    "DET",
    "SAS",
    "ATL",
    "GSW",
    "DEN",
    "SAC",
    "UTA",
    "OKC",
    "MIN",
]

idx_abbr = 0
df["team_abbr"] = ""
for i in range(len(df)):
    df.loc[i, "team_abbr"] = abbr[idx_abbr]
    if df.iloc[i].year == 2019 and df.iloc[i + 1].year == 2015:
        idx_abbr += 1

import os
from config import config

dir_out = os.path.join(config.get("dir", "raw"), "20200109")
os.makedirs(dir_out, exist_ok=True)
df.to_csv(os.path.join(dir_out, "historical.csv"), index=False)
