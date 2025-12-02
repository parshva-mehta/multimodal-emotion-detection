"""
RAVDESS Preprocessing Script (Raw Audio + Raw Video Frames, No Encoders)

This script converts raw RAVDESS audio (.wav) and video (.mp4) files into the
generic format expected by data.py's MultimodalDataset, WITHOUT any feature
encoding (encoders are handled later by encoders.py).

Output structure:

out_root/
    train/
        audio.npy    # (N_train, T_audio, 1)          raw waveform sequences
        video.npy    # (N_train, T_video, F_v)        grayscale frames flattened
        labels.npy   # (N_train,)
    val/
        ...
    test/
        ...

Assumptions:
    - audio_root contains all .wav files
    - video_root contains all .mp4 files (optional if using audio-only)
    - Every sample has a unique 7-part numeric filename stem
      e.g., '02-01-06-01-02-01-12.wav' / '.mp4'

Filename encoding (7 parts: A-B-C-D-E-F-G):

    A = Modality          (01 = full-AV, 02 = video-only, 03 = audio-only)
    B = Vocal channel     (01 = speech, 02 = song)
    C = Emotion           (01 = neutral, 02 = calm, 03 = happy, 04 = sad,
                           05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    D = Intensity         (01 = normal, 02 = strong, except neutral)
    E = Statement         (01 = "kids", 02 = "dogs")
    F = Repetition        (01 = 1st repetition, 02 = 2nd repetition)
    G = Actor             (01–24; odd = male, even = female)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import cv2
from sklearn.model_selection import train_test_split


# =============================
# Global hyperparameters
# =============================

# Audio: higher sampling rate & slightly longer duration (still CPU-manageable)
AUDIO_SR = 16000             # Hz
AUDIO_MAX_DURATION = 3.0     # seconds -> T_audio = AUDIO_SR * AUDIO_MAX_DURATION
# T_audio = 48000 at 16kHz / 3s

# Video: more frames & larger spatial size, but still safe on CPU
VIDEO_MAX_FRAMES = 24        # T_video
VIDEO_H = 64                 # height
VIDEO_W = 64                 # width
# frame_dim for FrameEncoder = VIDEO_H * VIDEO_W (grayscale) = 4096


# -----------------------------
# Filename parsing
# -----------------------------

def parse_ravdess_filename(fname: str) -> Dict[str, int]:
    """
    Parse RAVDESS filename to extract metadata.

    Input: filename with or without extension, e.g. '02-01-06-01-02-01-12.wav'

    Returns:
        dict with keys:
            'modality', 'channel', 'emotion', 'intensity',
            'statement', 'repetition', 'actor'
    """
    stem = Path(fname).stem  # strip extension
    parts = stem.split('-')
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {fname}")

    return {
        "modality": int(parts[0]),
        "channel": int(parts[1]),
        "emotion": int(parts[2]),
        "intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor": int(parts[6]),
    }


def map_emotion_label(meta: Dict[str, int]) -> int:
    """
    Map RAVDESS emotion code (01-08) to 0-based class index (0-7).

    Emotion codes:
        01 = neutral
        02 = calm
        03 = happy
        04 = sad
        05 = angry
        06 = fearful
        07 = disgust
        08 = surprised
    """
    emotion_code = meta["emotion"]
    if not (1 <= emotion_code <= 8):
        raise ValueError(f"Invalid emotion code: {emotion_code}")
    return emotion_code - 1


# -----------------------------
# File discovery
# -----------------------------

def load_filepaths(root_dir: str, ext: str) -> List[Path]:
    """
    Recursively list all files under root_dir with given extension.
    """
    root = Path(root_dir)
    return sorted(root.rglob(f"*{ext}"))


def build_stem_map(filepaths: List[Path]) -> Dict[str, Path]:
    """
    Build a mapping from filename stem (without extension) to full path.
    """
    stem_map = {}
    for fp in filepaths:
        stem = fp.stem
        if stem in stem_map:
            raise ValueError(f"Duplicate stem found: {stem} for {fp} and {stem_map[stem]}")
        stem_map[stem] = fp
    return stem_map


def build_join_key_map(filepaths: List[Path]) -> Dict[str, Path]:
    """Map RAVDESS files by last 6 fields
    (channel–emotion–intensity–statement–repetition–actor).

    This lets us align audio and video even if the modality code (first field)
    differs between them.

    If multiple files share the same join key (e.g., modality 01 and 02),
    we keep a single one using the preference: 01 (full AV) > 02 (video-only) > 03 (audio-only).
    """
    join_map: Dict[str, Path] = {}

    # modality preference: lower rank = better
    modality_rank = {1: 0, 2: 1, 3: 2}

    for fp in filepaths:
        parts = fp.stem.split("-")
        if len(parts) != 7:
            raise ValueError(f"Unexpected RAVDESS filename format: {fp}")
        modality = int(parts[0])
        join_key = "-".join(parts[1:])  # '01-06-01-02-01-12', etc.

        if join_key not in join_map:
            join_map[join_key] = fp
        else:
            # Decide whether to replace existing choice for this key
            existing_fp = join_map[join_key]
            existing_modality = int(existing_fp.stem.split("-")[0])

            # Look up rank, default to lowest priority if unknown
            existing_rank = modality_rank.get(existing_modality, 999)
            new_rank = modality_rank.get(modality, 999)

            # Keep the better (lower rank) modality
            if new_rank < existing_rank:
                join_map[join_key] = fp

    return join_map


# -----------------------------
# Raw audio extraction
# -----------------------------

def load_raw_audio(
    wav_path: Path,
    sr: int = AUDIO_SR,
    max_duration: float = AUDIO_MAX_DURATION,
) -> np.ndarray:
    """
    Load raw audio waveform, resample, and truncate/pad to fixed length.

    Returns:
        waveform: np.ndarray of shape (T_audio, 1)
                  where T_audio = sr * max_duration
    """
    y, _ = librosa.load(str(wav_path), sr=sr)

    max_len = int(max_duration * sr)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        pad = max_len - len(y)
        if pad > 0:
            y = np.pad(y, (0, pad), mode="constant")

    # normalize (optional)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    y = y.astype(np.float32)
    y = y.reshape(-1, 1)  # (T_audio, 1)
    return y


# -----------------------------
# Raw video frame extraction (grayscale, flattened)
# -----------------------------

def load_raw_video_frames(
    video_path: Path,
    max_frames: int = VIDEO_MAX_FRAMES,
    frame_height: int = VIDEO_H,
    frame_width: int = VIDEO_W,
) -> np.ndarray:
    """
    Extract raw grayscale frames from a video, uniformly sampled, resized,
    flattened per frame.

    Returns:
        frames: np.ndarray of shape (T_video, F_v)
                where T_video = max_frames
                      F_v = frame_height * frame_width
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        # fallback to zeros if no frames
        return np.zeros((max_frames, frame_height * frame_width), dtype=np.float32)

    num_frames = len(frames)
    # Sample indices
    if num_frames >= max_frames:
        indices = np.linspace(0, num_frames - 1, max_frames).astype(int)
    else:
        indices = np.arange(num_frames)

    feat_frames = []
    for idx in indices:
        frame = frames[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1] for stability
        gray = gray.astype(np.float32) / 255.0
        feat = gray.reshape(-1)  # (H*W,)
        feat_frames.append(feat)

    feat_frames = np.stack(feat_frames, axis=0)  # (T_used, F_v)

    # pad in time if needed
    T_used = feat_frames.shape[0]
    if T_used < max_frames:
        pad = np.zeros((max_frames - T_used, frame_height * frame_width), dtype=np.float32)
        feat_frames = np.concatenate([feat_frames, pad], axis=0)

    return feat_frames  # (max_frames, F_v)


# -----------------------------
# Dataset building
# -----------------------------

def build_ravdess_multimodal_raw(
    audio_root: str,
    video_root: Optional[str] = None,
    use_video: bool = True,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_by: str = "emotion",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build multimodal (raw audio + raw frames) RAVDESS dataset.

    Args:
        audio_root: directory containing all .wav files.
        video_root: directory containing all .mp4 files (optional if use_video=False).
        use_video: whether to include video modality.
        val_size: fraction for validation split.
        test_size: fraction for test split.
        random_state: random seed.
        stratify_by: 'emotion', 'actor', or None.

    Returns:
        (train_data, val_data, test_data) where each is:
            train_data = {
                'audio': (N, T_audio, 1),
                'video': (N, T_video, F_v)  [if use_video],
                'labels': (N,)
            }
    """
    # Discover files
    audio_files = load_filepaths(audio_root, ext=".wav")
    if len(audio_files) == 0:
        raise RuntimeError(f"No .wav files found under {audio_root}")

    # Use join keys (fields 2–7) so modality code can differ
    audio_map = build_join_key_map(audio_files)

    if use_video:
        if video_root is None:
            raise ValueError("use_video=True but video_root is None")
        video_files = load_filepaths(video_root, ext=".mp4")
        if len(video_files) == 0:
            raise RuntimeError(f"No .mp4 files found under {video_root}")
        video_map = build_join_key_map(video_files)

        common_keys = sorted(set(audio_map.keys()) & set(video_map.keys()))
        if len(common_keys) == 0:
            # debug: show a couple of example keys to help if paths are wrong
            example_audio = list(audio_map.keys())[:5]
            example_video = list(video_map.keys())[:5]
            raise RuntimeError(
                "No matching keys between audio and video sets.\n"
                f"Example audio keys: {example_audio}\n"
                f"Example video keys: {example_video}"
            )
        print(f"Found {len(common_keys)} matched audio+video samples.")
    else:
        common_keys = sorted(audio_map.keys())
        print(f"Using audio only; found {len(common_keys)} audio samples.")

    audio_feats = []
    video_feats = [] if use_video else None
    labels = []
    strat_keys = []

    # Extract raw sequences
    for key in common_keys:
        audio_path = audio_map[key]
        meta = parse_ravdess_filename(audio_path.name)
        label = map_emotion_label(meta)

        # raw audio
        wav_seq = load_raw_audio(audio_path)  # (T_audio, 1)
        audio_feats.append(wav_seq)

        # raw frames
        if use_video:
            video_path = video_map[key]
            frames_seq = load_raw_video_frames(video_path)  # (T_video, F_v)
            video_feats.append(frames_seq)

        labels.append(label)
        if stratify_by == "emotion":
            strat_keys.append(label)
        elif stratify_by == "actor":
            strat_keys.append(meta["actor"])
        else:
            strat_keys.append(0)

    audio_feats = np.stack(audio_feats, axis=0)          # (N, T_audio, 1)
    labels = np.array(labels, dtype=np.int64)            # (N,)
    strat_keys = np.array(strat_keys)

    if use_video:
        video_feats = np.stack(video_feats, axis=0)      # (N, T_video, F_v)

    print(f"Audio feats shape: {audio_feats.shape}")
    if use_video:
        print(f"Video feats shape: {video_feats.shape}")
    print(f"Labels shape: {labels.shape}")

    # Splits
    test_val_frac = val_size + test_size
    if not (0 < test_val_frac < 1):
        raise ValueError("val_size + test_size must be in (0, 1).")

    stratifier = strat_keys if stratify_by else None

    # First split: train vs temp
    X_a_train, X_a_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        audio_feats, labels, strat_keys,
        test_size=test_val_frac,
        random_state=random_state,
        stratify=stratifier,
    )

    if use_video:
        X_v_train, X_v_temp, _, _, _, _ = train_test_split(
            video_feats, labels, strat_keys,
            test_size=test_val_frac,
            random_state=random_state,
            stratify=stratifier,
        )

    # Second split: temp -> val/test
    val_frac_rel = val_size / (val_size + test_size)
    X_a_val, X_a_test, y_val, y_test = train_test_split(
        X_a_temp, y_temp,
        test_size=1 - val_frac_rel,
        random_state=random_state,
        stratify=y_temp if stratify_by else None,
    )

    if use_video:
        X_v_val, X_v_test, _, _ = train_test_split(
            X_v_temp, y_temp,
            test_size=1 - val_frac_rel,
            random_state=random_state,
            stratify=y_temp if stratify_by else None,
        )

    train_data: Dict[str, np.ndarray] = {
        "audio": X_a_train,
        "labels": y_train,
    }
    val_data: Dict[str, np.ndarray] = {
        "audio": X_a_val,
        "labels": y_val,
    }
    test_data: Dict[str, np.ndarray] = {
        "audio": X_a_test,
        "labels": y_test,
    }

    if use_video:
        train_data["video"] = X_v_train
        val_data["video"] = X_v_val
        test_data["video"] = X_v_test

    print(f"Train: {X_a_train.shape[0]}, Val: {X_a_val.shape[0]}, Test: {X_a_test.shape[0]}")
    return train_data, val_data, test_data


# -----------------------------
# Saving to disk
# -----------------------------

def save_splits_to_disk(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    out_root: str,
    modalities: Optional[List[str]] = None,
):
    """
    Save splits into structure expected by MultimodalDataset in data.py.

    Args:
        train_data/val_data/test_data:
            dict with keys:
                - modality names ('audio', 'video', etc.)
                - 'labels'
        out_root: Output root directory.
        modalities: explicit list of modalities to save.
                    If None, inferred as all keys except 'labels'.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if modalities is None:
        modalities = [k for k in train_data.keys() if k != "labels"]

    def _save_split(split_name: str, data_dict: Dict[str, np.ndarray]):
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for m in modalities:
            if m not in data_dict:
                raise KeyError(f"Modality '{m}' missing from {split_name} data.")
            np.save(split_dir / f"{m}.npy", data_dict[m])

        np.save(split_dir / "labels.npy", data_dict["labels"])

    _save_split("train", train_data)
    _save_split("val", val_data)
    _save_split("test", test_data)

    print(f"Saved preprocessed data to: {out_root}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess RAVDESS (raw audio + grayscale frames) into MultimodalDataset format."
    )
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Path to folder containing all RAVDESS .wav files.")
    parser.add_argument("--video_root", type=str, default=None,
                        help="Path to folder containing all RAVDESS .mp4 files.")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output directory for preprocessed .npy files.")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of data used for validation.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fraction of data used for test.")
    parser.add_argument("--no_video", action="store_true",
                        help="If set, ignore video and preprocess audio only.")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Disable stratified splitting (default: stratify by emotion).")

    args = parser.parse_args()

    use_video = not args.no_video
    stratify_by = None if args.no_stratify else "emotion"

    train_data, val_data, test_data = build_ravdess_multimodal_raw(
        audio_root=args.audio_root,
        video_root=args.video_root,
        use_video=use_video,
        val_size=args.val_size,
        test_size=args.test_size,
        stratify_by=stratify_by,
    )

    modalities = ["audio", "video"] if use_video else ["audio"]

    save_splits_to_disk(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        out_root=args.out_root,
        modalities=modalities,
    )

    print("RAVDESS raw preprocessing complete.")
