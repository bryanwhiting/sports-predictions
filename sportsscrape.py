import os
import glob
import itertools

import pandas as pd
import prefect
from prefect import unmapped
from prefect import task, Flow, Parameter
from prefect.triggers import any_successful
from sportsreference.nba.teams import Teams
from sportsreference.nba.schedule import Schedule

from config import config


@task
def get_list_of_team_abbrs():
    try:
        teams = Teams()
        teams_df = teams.dataframes
        abbr = list(teams_df.abbreviation.values)
    except Exception as ex:
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
            "CHI",
            "NYK",
            "BRK",
            "ORL",
            "BOS",
            "CLE",
        ]
        print(ex)
    return sorted(abbr)


@task
def create_year_abbr_combo(abbrs, from_year, thru_year):
    years = list(range(from_year, thru_year + 1))
    years_abbr = list(itertools.product(years, abbrs))
    return years_abbr  # [0:5]


@task
def return_schedule_dataframe(year_abbr: tuple, dir_out: str):
    """TODO: this 'should' be a sqlite databse, but don't got time for dat
    https://docs.prefect.io/core/tutorials/advanced-mapping.html#reusability
    """
    logger = prefect.context.get("logger")
    year, abbr = year_abbr
    logger.info("Looping: %s, %s", abbr, year)
    sched = Schedule(abbr, year=year)
    df_sched = sched.dataframe
    df_sched["team_abbr"] = abbr
    df_sched["year"] = year

    fp_out = os.path.join(dir_out, f"{year}-{abbr}.csv")
    df_sched.to_csv(fp_out, index=False)
    return (year, abbr, df_sched)


# Triggers: https://docs.prefect.io/core/concepts/execution.html#triggers
# run the result if any of the above mapped results are successful
def concat_all_csv(dir_out, filename="yr2020"):
    df = pd.concat(map(pd.read_csv, glob.glob(f"{dir_out}/*.csv")))
    fp_out = os.path.join(dir_out, f"../{filename}.csv")
    df.to_csv(fp_out, index=False)


# Crate some custom tasks
task_concat = task(concat_all_csv, trigger=any_successful)

with Flow("Raw Pull") as flow_pull:

    dir_out = Parameter("dir_out")
    from_year = Parameter("from_year")
    thru_year = Parameter("thru_year")

    abbrs = get_list_of_team_abbrs()
    year_abbrs = create_year_abbr_combo(abbrs, from_year=from_year, thru_year=thru_year)
    dfs = return_schedule_dataframe.map(year_abbrs, dir_out=unmapped(dir_out))
    concat = task_concat(dir_out, upstream_tasks=[dfs])


if __name__ == "__main__":

    dir_out = os.path.join(config.get("dir", "raw"), config.get("date", "today"))
    os.makedirs(dir_out, exist_ok=True)
    state = flow_pull.run(
        parameters={"dir_out": dir_out, "from_year": 2020, "thru_year": 2020}
    )
    # TODO: run this task after [dfs], regardless if it failed
    # flow_pull.visualize()
    flow_pull.visualize(flow_state=state)

    print("Failed States:\n")
    df_state = state.result[dfs]  # list of State objects
    for state in df_state.map_states:
        if state.is_failed():
            # the cached_inputs are the function inputs, easily retrieved
            print(state.cached_inputs["year_abbr"].value)
