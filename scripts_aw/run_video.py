"""Predict brain responses to one video with TRIBE v2.

Videos, paths, checkpoint and default language are set in config.py next to this file;
pass the name of a video variable there (e.g. sherlock1), not a path.
Run on a GPU node with the tribev2 environment active (or submit run_video.sbatch):

    python scripts_aw/run_video.py sherlock1
    python scripts_aw/run_video.py sherlock1 --language french --overwrite

Writes to config.OUTPUT_ROOT / <name> (e.g. output/sherlock1/):

    preds.npy           predicted activity, (n_timesteps, n_vertices) on fsaverage5, one row per second
    segment_starts.npy  start time (s) in the video of each row of preds
    events.csv          events fed to the model (video/audio chunks and transcribed words)
    run_info.json       video, language, checkpoint, preds shape, GPU, Slurm job id, finish time

Also writes <video>.wav (neuralset's ExtractAudioFromVideo) and <video>.tsv (the whisperx
transcript) next to the video, so its folder must be writable. Both are reused on reruns.

The spaCy model for the language must be installed first (the check below says how);
otherwise neuralset pip-installs it mid-job.
"""

import json
import os
import time
from pathlib import Path

import fire

import config

# huggingface_hub reads HF_HOME when first imported, so set it before tribev2 is imported.
# Both are inherited by the whisperx subprocess (its models: HF_HOME; alignment model: TORCH_HOME).
os.environ["HF_HOME"] = str(config.HF_HOME)
os.environ["TORCH_HOME"] = str(config.TORCH_HOME)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spacy.util  # noqa: E402
import torch  # noqa: E402
from neuralset.events.transforms import (  # noqa: E402
    AddContextToWords,
    AddSentenceToWords,
    AddText,
    ChunkEvents,
    ExtractAudioFromVideo,
    RemoveMissing,
)
from neuralset.events.utils import standardize_events  # noqa: E402

from tribev2.demo_utils import VALID_SUFFIXES, TribeModel  # noqa: E402
from tribev2.eventstransforms import ExtractWordsFromAudio  # noqa: E402

# spaCy model neuralset's sentence step loads per language (neuralset.utils.get_spacy_model).
# These languages are also accepted by the whisperx call (tribev2/eventstransforms.py);
# whisperx's dutch has no spaCy model there, so it isn't supported end to end.
SPACY_MODELS = {
    "english": "en_core_web_lg",
    "french": "fr_core_news_lg",
    "spanish": "es_core_news_lg",
    "chinese": "zh_core_web_lg",
}


def build_events(video: Path, language: str) -> pd.DataFrame:
    """Same as TribeModel.get_events_dataframe(video_path=...), with a choice of speech language.

    Mirrors tribev2.demo_utils.get_audio_and_text_events step for step; only the
    ExtractWordsFromAudio language differs. Keep in sync if that function changes.
    """
    event = {
        "type": "Video",
        "filepath": str(video),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
        ExtractWordsFromAudio(language=language),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.05),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    events = standardize_events(pd.DataFrame([event]))
    for transform in transforms:
        events = transform(events)
    return standardize_events(events)


def main(vid_path: str, language: str = config.LANGUAGE, overwrite: bool = False) -> None:
    """Predict brain responses to a video and save them under config.OUTPUT_ROOT / vid_path.

    Args:
        vid_path: name of the video's variable in config.py (e.g. sherlock1).
        language: language of the speech in the video (english, french, spanish, chinese).
        overwrite: rerun even if this video already has predictions.
    """
    # Videos are the lowercase Path variables in config.py; settings are UPPERCASE.
    videos = {k: v for k, v in vars(config).items() if isinstance(v, Path) and k.islower()}
    vid_path = str(vid_path)  # fire turns numeric-looking names into ints
    if vid_path not in videos:
        raise ValueError(
            f"Unknown video {vid_path!r}: add `{vid_path} = Path(...)` to config.py. "
            f"Videos there: {sorted(videos)}"
        )
    video = videos[vid_path].expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.suffix.lower() not in VALID_SUFFIXES["video_path"]:
        raise ValueError(
            f"Video must end with one of {sorted(VALID_SUFFIXES['video_path'])}: {video}"
        )
    already_written = all(video.with_suffix(s).exists() for s in (".wav", ".tsv"))
    if not already_written and not os.access(video.parent, os.W_OK):
        raise PermissionError(
            f"{video.parent} isn't writable; the audio ({video.stem}.wav) and transcript "
            f"({video.stem}.tsv) are written next to the video"
        )
    if language not in SPACY_MODELS:
        raise ValueError(f"language must be one of {tuple(SPACY_MODELS)}, got {language!r}")
    if not spacy.util.is_package(SPACY_MODELS[language]):
        raise RuntimeError(
            f"spaCy model {SPACY_MODELS[language]} isn't installed; install it once with the env active:\n"
            f"    python -m spacy download {SPACY_MODELS[language]} --no-cache-dir"
        )

    out_dir = config.OUTPUT_ROOT / vid_path
    if (out_dir / "preds.npy").exists() and not overwrite:
        print(f"{out_dir / 'preds.npy'} already exists; pass --overwrite to redo it")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TribeModel.from_pretrained(
        config.CHECKPOINT, cache_folder=config.FEATURE_CACHE, device=config.DEVICE
    )
    events = build_events(video, language)
    events.to_csv(out_dir / "events.csv", index=False)

    preds, segments = model.predict(events=events)
    np.save(out_dir / "preds.npy", preds)
    np.save(out_dir / "segment_starts.npy", np.array([s.start for s in segments]))

    run_info = {
        "name": vid_path,
        "video": str(video),
        "language": language,
        "checkpoint": config.CHECKPOINT,
        "preds_shape": list(preds.shape),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Saved preds {preds.shape} (n_timesteps, n_vertices) to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
