import sys
import os
import multiprocessing
from pathlib import Path
# get parent of script directory (app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from components.tools import utils

parameters_file = sys.argv[1]

try:
    
    config = utils.read_toml(Path(parameters_file))

    cpu_limit = config.get('Config', {}).get('CPU count limit', 'ncpus')
    
    if cpu_limit == 'ncpus':
        cpu_count = multiprocessing.cpu_count()
    else:
        try:
            cpu_count = int(cpu_limit)
            cpu_count = max(1, min(cpu_count, multiprocessing.cpu_count()))
        except (ValueError, TypeError):
            cpu_count = multiprocessing.cpu_count() - 1
    
    print(cpu_count)
except Exception as e:
    print(4, file=sys.stderr)  # fallback
    print(4)