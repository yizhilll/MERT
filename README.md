# MERT

This is the official implementation of the paper "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training".


**Evaluation, Benchmarking and Baselines**:
* The codes for downstream task evaluation on MERT and baseline models can be referred to [MARBLE](https://marble-bm.shef.ac.uk) benchmark.
* MERT is also evaluated with the [MARBLE protocol](https://marble-bm.shef.ac.uk/submit) and reported on the [music understanding leaderboard](https://marble-bm.shef.ac.uk/leaderboard). 

## Training

The MERT training is implemented with [fairseq](https://github.com/facebookresearch/fairseq). 
You need to clone the fairseq repo inside our repo at `./src/fairseq` and MERT implementation codes as a fairseq example projcet. 

### Environment Setup

The training of MERT requires:
* [fairseq](https://github.com/facebookresearch/fairseq) & [pytorch](https://pytorch.org/) for the training (must)
* [nnAudio](https://github.com/KinWaiCheuk/nnAudio) for on-the-fly CQT inference (must)
* [apex](https://github.com/NVIDIA/apex) for half-precision training (optaional)
* [nccl](https://github.com/NVIDIA/nccl) for multiple device training (optional)
* [fairscale](https://github.com/facebookresearch/fairscale) for FSDP and CPU offloading (optional)
* [WARNING] the version of [transformers](https://huggingface.co/docs/transformers/en/index) requires to be `transformers==4.38` since a latter update cause incompatible.

  
You could use the script `./scripts/environment_setup.sh` to set up the python environment from scarth, which could be easily modified to DOCKERFILE. 
All the relevant folders will be placed at the customized MERT repo folder path `$MAP_PROJ_DIR`.

### Data Preparation

Generally, there are 2 things you need to prepare:
* `DATA_DIR=${MAP_PROJ_DIR}/data/audio_tsv`: a folder that contains a `train.tsv` and a `valid.tsv` file, which specify the root path to the audios at the first line and the relative paths at the rest lines.
* `LABEL_ROOT_DIR=${MAP_PROJ_DIR}/data/labels`: a folder filled with all the discrete tokens that need to prepare before training. They could be K-means or RVQ-VAE tokens.

The two options for acoustic teacher peuso labels in MERT training can be constructed by:
* K-means Labels from [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans) (the vanilla MFCC version)
* codecs from [EnCodec](https://github.com/facebookresearch/encodec)

Scripts for preparing the training data:
```shell
# First prepare the manifest file indexing the audios.
# If needed the audios will be converted to 24K Hz.
python scripts/prepare_manifest.py --root-dir /absolute/path/to/original/custom_audio_dataset \
      --target-rate 24000 --converted-root-dir /absolute/path/to/converted/custom_audio_dataset \
      --out-dir data/custom_audio_dataset_manifest --extension wav
      
# Prepare the codecs for audios in the manifest
python scripts/prepare_codecs_from_manifest.py  \
      --manifest_path data/custom_audio_dataset_manifest --manifest_file_name train.tsv \
      --out_root data/encodec_labels/custom_audio_dataset --codebook_size 1024 --n_codebook 8
```

The data preparation and format can be referred to [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) for more details.

### Start Training

Noted that we follow the fariseq development protocol to put our codes as an example project. 
When running the fairseq program, you can specify the MERT customized codes by `common.user_dir=${MAP_PROJ_DIR}/mert_faiseq`.


After the environment is set up, you could use the following scripts:
```shell
# for MERT95M
bash scripts/run_training.sh 0 dummy MERT_RVQ-VAE_CQT_95M

# for MERT 330M
bash scripts/run_training.sh 0 dummy MERT_RVQ-VAE_CQT_330M
```

## Inference

We use the huggingface models for interface and evaluation. Using the example of RVQ-VAE 95M MERT as example, the following codes show how to load and extract representations with MERT.

```shell
python MERT/scripts/MERT_demo_inference.py
```

## Checkpoints

### Huggingface Checkpoint

Our Huggingface Transformers checkpoints for convenient inference are uploaded to the [m-a-p](https://huggingface.co/m-a-p) project page.
* [MERT-v0](https://huggingface.co/m-a-p/MERT-v1-95M): The base (95M) model trained with K-means acoustic teacher and musical teacher.
* [MERT-v0-public](https://huggingface.co/m-a-p/MERT-v1-95M): The base (95M) model trained with K-means acoustic teacher and musical teacher using the public music4all training data.
* [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M): The base (95M) model trained with RVQ-VAE acoustic teacher and musical teacher.
* [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M): The large (330M) model trained with RVQ-VAE acoustic teacher and musical teacher.

To convert your self-trained models, check the scripts:
```shell
bash scripts/convert_HF_script.sh default mert config_mert_base [/absolute/path/to/a/fairseq/checkpoint.pt]
```

### Fairseq Checkpoint

We also provide the corresponding fairseq checkpoint for continual training or further modification hosted at the corresponding HF repos:
* [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/MERT-v1-95M_fairseq.pt) 
* [MERT-v1-330M](https://huggingface.co/m-a-p/MERT-v1-330M/blob/main/MERT-v1-330M_fairseq.pt) 


## Citation

```shell
@misc{li2023mert,
      title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training}, 
      author={Yizhi Li and Ruibin Yuan and Ge Zhang and Yinghao Ma and Xingran Chen and Hanzhi Yin and Chenghua Lin and Anton Ragni and Emmanouil Benetos and Norbert Gyenge and Roger Dannenberg and Ruibo Liu and Wenhu Chen and Gus Xia and Yemin Shi and Wenhao Huang and Yike Guo and Jie Fu},
      year={2023},
      eprint={2306.00107},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
