from pathlib import Path
import json
import sys
from pathlib import Path as P

# Ensure workspace root is importable
sys.path.insert(0, str(P.cwd()))
from scripts.pilot_model import run_pilot_for_model


if __name__ == '__main__':
    res = run_pilot_for_model('SVM', P.cwd() / 'config.yaml', workers=1, model_threads=1)
    # print minimal summary
    print(json.dumps({'run_id': res['run_id'], 'total_candidates': res['total_candidates'], 'total_success': res['total_success'], 'total_failed': res['total_failed'], 'total_elapsed_seconds': res['total_elapsed_seconds']}, indent=2))
