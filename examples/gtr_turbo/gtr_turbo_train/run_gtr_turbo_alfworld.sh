#!/bin/bash
# GTR-Turbo training on ALFWorld environment.
#
# Usage:
#   bash examples/gtr_turbo/gtr_turbo_train/run_gtr_turbo_alfworld.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SLIME_SCRIPT_NUM_GPUS="${SLIME_SCRIPT_NUM_GPUS:-8}"

# Create ALFWorld-specific config by overriding env settings
python -c "
import yaml
config = yaml.safe_load(open('${SCRIPT_DIR}/config.yaml'))
config['env'] = 'alfworld'
config['env_config'] = 'examples/gtr_turbo/alfworld/config.yaml'
config['save_dir'] = '/root/outputs/gtr_turbo_alfworld'
yaml.dump(config, open('/tmp/gtr_turbo_alfworld_config.yaml', 'w'))
"

python -m examples.gtr_turbo.gtr_turbo_train.train_gtr_turbo \
    --config /tmp/gtr_turbo_alfworld_config.yaml
