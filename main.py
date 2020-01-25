"""


"""

import fire
from config import config
import os
from sportsscrape import flow_pull
from dataproc import flow_proc
from output import flow_output

def pull(start=2020, end=2020):
    """Data pull. Specify start and end year."""
    dir_out = os.path.join(config.get("dir", "raw"), config.get("date", "today"))
    os.makedirs(dir_out, exist_ok=True)
    state = flow_pull.run(
        parameters={"dir_out": dir_out, "from_year": start, "thru_year": end}
    )


def proc(nrows: int = None, past: int = None ):
    """Process the pulled data

    Process historical data: 
    python main.py proc --past 1 
    
    Process 2020 data:
    python main.py proc
    """
    if past == 1:
        filename = 'historical'
    else:
        filename = 'yr2020'
    state = flow_proc.run(parameters={"nrows": nrows, "filename": filename})

def results():
    flow_output.run()


def run(d=1, p=1, r=1):
    """Do a daily run
    d: int 
        download
    p: process
    r: output - scores and gets accuracy 
    """
    if d==1:
        pull()
    if p == 1: 
        proc()
    if r == 1:
        results()

if __name__ == "__main__":
    # python flows.py pull --thru-year
    fire.Fire(
        {"pull": pull, "proc": proc, 'results': results, 'run': run}
    )
