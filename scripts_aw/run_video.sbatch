#!/bin/bash
#SBATCH --job-name=tribev2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ceph/behrens/awong/tribev2_analyses/output/logs/%x_%j.out
#
# Predict brain responses to one video with TRIBE v2 (settings in scripts_aw/config.py).
#
#   sbatch scripts_aw/run_video.sbatch sherlock1
#   sbatch scripts_aw/run_video.sbatch sherlock2 --language french
#
# The name is a video variable in scripts_aw/config.py, where its path is set.
#
# Everything after the script name is passed to run_video.py. Slurm won't create the
# log folder above, so make it once first:
#   mkdir -p /ceph/behrens/awong/tribev2_analyses/output/logs

# uvload is an alias from ~/.bashrc; bash only expands aliases in scripts with expand_aliases on.
shopt -s expand_aliases
source ~/.bashrc
uvload && source /ceph/python_envs/$USER/tribev2/bin/activate

set -euo pipefail

if [[ "${UV_CACHE_DIR:-}" != "/ceph/python_envs/$USER/.uv_cache" ||
      "${VIRTUAL_ENV:-}" != "/ceph/python_envs/$USER/tribev2" ]]; then
    echo "Environment setup failed: uvload or the tribev2 env did not load." >&2
    echo "Check that ~/.bashrc defines uvload and doesn't stop early for batch jobs." >&2
    exit 1
fi

# Slurm runs a copy of this file, so the repo location can't be worked out from $0.
REPO=/ceph/behrens/awong/tribev2

nvidia-smi
python "$REPO/scripts_aw/run_video.py" "$@"
