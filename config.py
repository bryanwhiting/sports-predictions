import os
from configparser import ConfigParser
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')

# Create the config
config = ConfigParser(default_section='default')
dir_root = os.path.expanduser('~/github/sports-predictions')
config.read(os.path.join(dir_root, 'rsc/config.cfg'))

# Dates
config.add_section('date')
config.set('date', 'today', today)

