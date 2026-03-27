#!/bin/bash
set -e
source ~/.bashrc

cd /root/Training/ki/openpi
source ./.venv/bin/activate

echo "cpu 核心数目"
nproc

uv run scripts/compute_norm_stats.py --config-name pi05_libero_spatial_torch_debug
