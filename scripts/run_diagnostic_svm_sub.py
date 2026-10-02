import sys
from pathlib import Path
import json
import yaml

# Ensure workspace root importable
sys.path.insert(0, str(Path.cwd()))

from scripts.pilot_model import run_pilot_for_model

# Accept config path arg or use default
cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / 'config.yaml'
# Load config and override
with cfg_path.open('r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('experiment', {})['workers'] = 1
cfg.setdefault('experiment', {})['threads'] = 1
cfg.setdefault('gwo', {})['population_size'] = 5
cfg.setdefault('gwo', {})['iterations'] = 1
cfg.setdefault('models', {})['enabled'] = ['SVM']

# write temp config
run_dir = Path.cwd() / 'runs' / f"diagnostic_svm_{int(__import__('time').time())}"
run_dir.mkdir(parents=True, exist_ok=True)
cfg_file = run_dir / 'config_used.yaml'
with cfg_file.open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f)

res = run_pilot_for_model('SVM', cfg_file, workers=1, model_threads=1)
print(json.dumps({'run_id': res['run_id'], 'total_candidates': res['total_candidates'], 'total_success': res['total_success'], 'total_failed': res['total_failed'], 'total_elapsed_seconds': res['total_elapsed_seconds']}, indent=2))
