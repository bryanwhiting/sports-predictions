"""


"""

import fire
from config import config
import os
from sportsscrape import flow_pull
from dataproc import flow_proc
from output import flow_output
from rmarkdown import flow_rmd


def pull(start=2020, end=2020):
    """Data pull. Specify start and end year."""
    dir_out = os.path.join(config.get("dir", "raw"), config.get("date", "today"))
    os.makedirs(dir_out, exist_ok=True)
    flow_pull.run(
        parameters={"dir_out": dir_out, "from_year": start, "thru_year": end}
    )


def proc(nrows: int = None, past: int = None):
    """Process the pulled data

    Process historical data:
    python main.py proc --past 1

    Process 2020 data:
    python main.py proc
    """
    if past == 1:
        filename = "historical"
    else:
        filename = "yr2020"
    flow_proc.run(parameters={"nrows": nrows, "filename": filename})


def results():
    flow_output.run()


def rmarkdown():
    flow_rmd.run()


def run(d=1, p=1, o=1, r=1):
    """Do a daily run
    d: int
        download
    p: process
    o: output - scores and gets accuracy
    r: rmarkdown
    """
    if d == 1:
        pull()
    if p == 1:
        proc()
    if o == 1:
        results()
    if r == 1:
        rmarkdown()


if __name__ == "__main__":
    # python main.py run
    fire.Fire(
        {"pull": pull, "proc": proc, "results": results, "rmd": rmarkdown, "run": run}
    )
