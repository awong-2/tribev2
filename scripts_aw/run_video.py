"""Predict brain responses to one video with TRIBE v2.

Paths, checkpoint and default language are set in config.py next to this file.
Run on a GPU node with the tribev2 environment active (or submit run_video.sbatch):

    python scripts_aw/run_video.py /path/to/video.mp4
    python scripts_aw/run_video.py /path/to/video.mp4 --language french --out_name clip01 --overwrite

Writes to config.OUTPUT_ROOT / <out_name, or the video file name without extension>:

    preds.npy           predicted activity, (n_timesteps, n_vertices) on fsaverage5, one row per second
    segment_starts.npy  start time (s) in the video of each row of preds
    events.csv          events fed to the model (video/audio chunks and transcribed words)
    run_info.json       video, language, checkpoint, preds shape, GPU, Slurm job id, finish time
"""

import json
import os
import time
from pathlib import Path

import fire

import config

# huggingface_hub reads HF_HOME when first imported, so set it before tribev2 is imported.
os.environ["HF_HOME"] = str(config.HF_HOME)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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

# Languages the whisperx call accepts, see tribev2/eventstransforms.py (_get_transcript_from_audio).
LANGUAGES = ("english", "french", "spanish", "dutch", "chinese")


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


def main(
    video: str,
    language: str = config.LANGUAGE,
    out_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Predict brain responses to a video and save them under config.OUTPUT_ROOT.

    Args:
        video: path to the video file.
        language: language of the speech in the video (english, french, spanish, dutch, chinese).
        out_name: output folder name; defaults to the video file name without extension.
        overwrite: rerun even if this output folder already has predictions.
    """
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.suffix.lower() not in VALID_SUFFIXES["video_path"]:
        raise ValueError(
            f"Video must end with one of {sorted(VALID_SUFFIXES['video_path'])}: {video}"
        )
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")

    # str() because fire turns numeric-looking names like 01 into ints.
    out_dir = config.OUTPUT_ROOT / (str(out_name) if out_name is not None else video.stem)
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
