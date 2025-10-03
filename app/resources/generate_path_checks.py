import sys
import os
from pathlib import Path

# get parent of script directory (app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from components.tools import utils

parameters_file = sys.argv[1]
checks_file = sys.argv[2]

parameters = utils.read_toml(Path(parameters_file))
new_checks = []
for key, value in parameters['Data paths'].items():
    if key == 'Database file':
        value = os.path.join(*value)
        new_check = '\t'.join([
            value,
            f"{key} folder is missing. Mount host {key} directory to {value}. If database has not been created yet, run the updater container to create it.",
            f"{key} still not found; app will start but {key} features will fail."
        ])

if len(new_checks) > 0:
    with open(checks_file, 'a') as f:
        for check in new_checks:
            f.write(check)