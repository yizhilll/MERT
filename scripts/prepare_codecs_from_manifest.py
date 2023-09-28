import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob

import numpy as np
import torch

from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence

import torchaudio

from encodec import EncodecModel
from encodec.utils import convert_audio

class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--manifest_file_name', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--n_codebook', type=int, default=8)

    parsed = parser.parse_args()

    os.makedirs(parsed.out_root, exist_ok=True)
    return parsed


def main(args: Namespace) -> None:

    model = EncodecModel.encodec_model_24khz()

    model.set_target_bandwidth(6.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    manifest_path=os.path.join(args.manifest_path, args.manifest_file_name)
    
    if 'train' in args.manifest_file_name:
        f_codecs = [open(os.path.join(args.out_root, f'train.codec_{i}'), 'w') for i in range(args.n_codebook)]
    else:
        f_codecs = [open(os.path.join(args.out_root, f'valid.codec_{i}'), 'w') for i in range(args.n_codebook)]

    with open(manifest_path, 'r') as f:
        root_dir = f.readline().strip()
        for line in tqdm(f):
            relative_path, n_sample = line.strip().split('\t')
            
            
            wav, sr = torchaudio.load(os.path.join(root_dir, relative_path))
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0)
            wav = wav.to(device)
            with torch.no_grad():
                encoded_frames = model.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            codes = codes.to('cpu')[0]

            for i in range(args.n_codebook):
                f_codecs[i].write(' '.join([str(x) for x in codes[i].numpy()]) + '\n')

    # close the codecs txt files
    for i in range(args.n_codebook):
        f_codecs[i].close()


    # clean the CUDA cache to save memory
    model.to('cpu')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main(parse_args())
