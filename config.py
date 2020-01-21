import os
import joblib
from configparser import ConfigParser
from datetime import datetime
import pandas
import xgboost

today = datetime.now().strftime("%Y-%m-%d")

# Create the config
config = ConfigParser(default_section="default")
dir_root = os.path.expanduser("~/github/sports-predictions")
config.read(os.path.join(dir_root, "rsc/config.cfg"))

# Dates
config.add_section("date")
config.set("date", "today", today)

# Drop columns from prediction
config.drop_cols = ["boxscore_index", "year", "date", "datetime", "team1", "team2"]


# OTHER UTILS------
def save(obj, dir, filename, main=True, date=True):
    """If today=True then it'll save it in a folder.
    Otherwise it'll save a copy to the main"""
    today = datetime.now().strftime("%Y-%m-%d")
    fn = filename + '.jb' 

    # Save out
    _dir = config.get('dir', dir)
    os.makedirs(_dir, exist_ok=True)
    if main:
        fp = os.path.join(_dir, fn)
        joblib.dump(obj, filename=fp, compress=3)

    if date:
        _dir = os.path.join(_dir, today)
        os.makedirs(_dir, exist_ok=True)
        fp = os.path.join(_dir, fn)
        joblib.dump(obj, filename=fp, compress=3)

def load(dir, filename, date:str=None):
    """Loads object using joblib"""
    if date:
        fp = os.path.join(dir, date, filename)
    else:
        fp = os.path.join(dir, filename)
    return joblib.load(fp)


