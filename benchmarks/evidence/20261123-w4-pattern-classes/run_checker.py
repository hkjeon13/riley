import sys
from pathlib import Path

import check_distribution_gpu as checker

if "--structured-holdout" in sys.argv:
    checker.API_CHECK = Path(__file__).with_name("pattern_probe.py")
raise SystemExit(checker.main())
