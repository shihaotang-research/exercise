"""Acute exercise-response module–phenotype associations from existing networks."""
import os
from pathlib import Path
from _module_phenotype_core import run

if __name__ == '__main__':
    run('acute', Path(os.environ.get('MULTIOMICS_DATA_DIR', os.environ.get('PROJECT_DIR', '.'))))
