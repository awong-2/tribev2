"""Settings for run_video.py. Edit here to change where things go."""

from pathlib import Path

# Everything TRIBE v2 writes (predictions, feature cache, HuggingFace downloads) lives under here.
ANALYSIS_ROOT = Path("/ceph/behrens/awong/tribev2_analyses")

OUTPUT_ROOT = ANALYSIS_ROOT / "output"  # one folder per video, plus logs/ for Slurm
FEATURE_CACHE = ANALYSIS_ROOT / "feature_cache"  # extracted features, reused across runs
HF_HOME = ANALYSIS_ROOT / "hf_cache"  # HuggingFace model downloads and login token

CHECKPOINT = "facebook/tribev2"
DEVICE = "auto"  # "cuda" when a GPU is available, else "cpu"
LANGUAGE = "english"  # default speech language; override per video with --language
