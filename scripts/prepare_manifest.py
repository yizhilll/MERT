# This source code is modified from
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
"""
Data pre-processing: create tsv files for training (and valiation).
"""
import logging
import re
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torchaudio

import argparse
from tqdm import tqdm

_LG = logging.getLogger(__name__)


def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    valid_percent: float = 0.01,
    seed: int = 1317,
    extension: str = "flac",
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        valid_percent (float, optional): The percentage of data for validation. (Default: 0.01)
        seed (int): The seed for randomly selecting the validation files.
        extension (str, optional): The extension of audio files. (Default: ``flac``)

    Returns:
        None
    """
    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(seed)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = open(out_dir / f"valid.tsv", "w") if valid_percent > 0 else None
    # search_pattern = '*.train_.*.$' # prepare for extra pattern
    with open(out_dir / f"train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        for fname in tqdm(root_dir.glob(f"**/*.{extension}")):
            if args.target_rate <= 0:
                try:
                    frames = torchaudio.info(fname).num_frames
                    dest = train_f if torch.rand(1) > valid_percent else valid_f
                    print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")
            else:
                # check the original sample rate, if not equal to target rate, convert it with torchaudio,
                # and save the converted file to the converted-root-dir in the same relative path
                try:
                    sr = torchaudio.info(fname).sample_rate
                    if sr != args.target_rate:
                        wav, sr = torchaudio.load(fname)
                        wav = torchaudio.functional.resample(wav, sr, args.target_rate)
                        dest = train_f if torch.rand(1) > valid_percent else valid_f
                        # save the converted file to the converted-root-dir in the same relative path
                        converted_fname = Path(os.path.join(args.converted_root_dir, fname.relative_to(root_dir)))
                        os.makedirs(os.path.dirname(converted_fname), exist_ok=True)
                        torchaudio.save(converted_fname, wav, args.target_rate)
                        frames = wav.shape[1] # = torchaudio.info(converted_fname).num_frames
                        print(f"{converted_fname.relative_to(args.converted_root_dir)}\t{frames}", file=dest)
                    else:
                        frames = torchaudio.info(fname).num_frames
                        dest = train_f if torch.rand(1) > valid_percent else valid_f
                        print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")
                    

    if valid_f is not None:
        valid_f.close()
    _LG.info("Finished creating the file lists successfully")

if __name__ == "__main__":

    # read arguments with argparse
    parser = argparse.ArgumentParser(description="Prepare manifest files for training")
    parser.add_argument("--root-dir", type=str, default="data/audio_folder", help="root dir of the audio files, must use absolute path")
    parser.add_argument("--converted-root-dir", type=str, default="data/audio_folder_converted", help="root dir of the new audio files folder, must use absolute path")
    parser.add_argument("--target-rate", type=int, default=-1, help="")
    parser.add_argument("--out-dir", type=str, default="data/audio_manifest")
    parser.add_argument("--valid-percent", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1317)
    parser.add_argument("--extension", type=str, default="flac")
    args = parser.parse_args()

    create_tsv(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        valid_percent=args.valid_percent,
        seed=args.seed,
        extension=args.extension,
    )

