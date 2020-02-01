import os
import pathlib
import joblib
from configparser import ConfigParser
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
dir_root = str(pathlib.Path(__file__).parent.absolute())

# Create the config
config = ConfigParser(default_section="default")
config.read(os.path.join(dir_root, "rsc/config.cfg"))

# Update
config.set("dir", "home", dir_root)
print("Home dir set to:", config.get("dir", "home"))

# Dates
config.add_section("date")
config.set("date", "today", today)

# Drop columns from prediction
config.drop_cols = ["boxscore_index", "year", "date", "datetime", "team1", "team2"]
config.targets = ["result", "pg_score1", "pg_score2", "pg_spread"]

# Season Model score columns


# OTHER UTILS------
def save(obj, dir, filename, ext=".jb", main=True, date=True):
    """If today=True then it'll save it in a folder.
    Otherwise it'll save a copy to the main"""
    today = datetime.now().strftime("%Y-%m-%d")
    fn = filename + ext

    # Save out
    _dir = config.get("dir", dir)
    os.makedirs(_dir, exist_ok=True)
    if main:
        fp = os.path.join(_dir, fn)
        if ext == ".jb":
            joblib.dump(obj, filename=fp, compress=3)
        else:
            obj.to_csv(fp, index=False)

    if date:
        _dir = os.path.join(_dir, today)
        os.makedirs(_dir, exist_ok=True)
        fp = os.path.join(_dir, fn)
        if ext == ".jb":
            joblib.dump(obj, filename=fp, compress=3)
        else:
            obj.to_csv(fp, index=False)

    print("Saved to ", fp)


def load(dir, filename, date: str = None):
    """Loads object using joblib"""
    _dir = config.get("dir", dir)
    fn = filename + ".jb"
    if date:
        fp = os.path.join(_dir, date, fn)
    else:
        fp = os.path.join(_dir, fn)
    print("Loading from:", fp)
    return joblib.load(fp)
