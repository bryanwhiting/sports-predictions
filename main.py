"""


"""

import fire
from config import config
import os
from sportsscrape import flow_pull
from dataproc import flow_proc


def pull(start=2020, end=2020):
    """Data pull
    from_year: int
    thru_year: int 
    """
    dir_out = os.path.join(config.get("dir", "raw"), config.get("date", "today"))
    os.makedirs(dir_out, exist_ok=True)
    state = flow_pull.run(
        parameters={"dir_out": dir_out, "from_year": start, "thru_year": end}
    )


def proc(nrows=None, filename="_historical.csv"):
    """Process the _historical data"""
    state = flow_proc.run(parameters={"nrows": nrows, "filename": filename})


if __name__ == "__main__":
    # python flows.py pull --thru-year
    fire.Fire(
        {"pull": pull, "proc": proc,}
    )
