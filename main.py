"""


"""

import fire
from config import config
import os
from sportsscrape import flow_pull
from dataproc import flow_proc


def pull(start=2020, end=2020):
    """Data pull. Specify start and end year."""
    dir_out = os.path.join(config.get("dir", "raw"), config.get("date", "today"))
    os.makedirs(dir_out, exist_ok=True)
    state = flow_pull.run(
        parameters={"dir_out": dir_out, "from_year": start, "thru_year": end}
    )


def proc(nrows: int = None, filename: str = "yr2020"):
    """Process the pulled data"""
    state = flow_proc.run(parameters={"nrows": nrows, "filename": filename})


if __name__ == "__main__":
    # python flows.py pull --thru-year
    fire.Fire(
        {"pull": pull, "proc": proc,}
    )
