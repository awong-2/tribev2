"""Settings for run_video.py. Edit here to change where things go."""

from pathlib import Path

# Everything TRIBE v2 writes (predictions, feature cache, model downloads) lives under here,
# except <video>.wav and <video>.tsv, which neuralset/tribev2 write next to each video.
ANALYSIS_ROOT = Path("/ceph/behrens/awong/tribev2_analyses")

OUTPUT_ROOT = ANALYSIS_ROOT / "output"  # one folder per video, plus logs/ for Slurm
FEATURE_CACHE = ANALYSIS_ROOT / "feature_cache"  # extracted features, reused across runs
HF_HOME = ANALYSIS_ROOT / "hf_cache"  # HuggingFace model downloads and login token
TORCH_HOME = ANALYSIS_ROOT / "torch_cache"  # PyTorch downloads (whisperx's alignment model)

CHECKPOINT = "facebook/tribev2"
DEVICE = "auto"  # "cuda" when a GPU is available, else "cpu"
LANGUAGE = "english"  # default speech language; override per video with --language

# Videos: pass the variable name to run_video.py (e.g. sherlock1); results go to OUTPUT_ROOT/<name>.
# Keep these lowercase (settings above are UPPERCASE). Fill in the real paths.
sherlock1 = Path("/path/to/sherlock_part1.mp4")
sherlock2 = Path("/path/to/sherlock_part2.mp4")
