# MERT

This is the official implementation of the paper "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training".


## Training

The MERT training is implemented with [fairseq](https://github.com/facebookresearch/fairseq). We clone the fairseq repo inside our repo and implement MERT as a fairseq example, and give the soft link of the code at `./mert_fairseq`.

### Environment Setup

The training of MERT requires:
* [fairseq](https://github.com/facebookresearch/fairseq) & [pytorch](https://pytorch.org/) for the training (must)
* [nnAudio](https://github.com/KinWaiCheuk/nnAudio) for on-the-fly CQT inference (must)
* [apex](https://github.com/NVIDIA/apex) for half-precision training (optaional)
* [nccl](https://github.com/NVIDIA/nccl) for multiple device training (optional)
* [fairscale](https://github.com/facebookresearch/fairscale) for FSDP and CPU offloading (optional)

You could use the script `./scripts/environment_setup.sh` to set up the python environment from scarth. All the relevant folders will be placed at the customized `$MAP_PROJ_DIR` folder.

### Data Preparation

The data preparation can be referred to [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) for more details.

Generally, there are 2 things you need to prepare:
* `DATA_DIR=${MAP_PROJ_DIR}/data/audio_tsv`: a folder that contains a `train.tsv` and a `valid.tsv` file, which specify the root path to the audios at the first line and the relative paths at the rest lines.
* `LABEL_ROOT_DIR=${MAP_PROJ_DIR}/data/labels`: a folder filled with all the discrete tokens that need to prepare before training. They could be K-means or RVQ-VAE tokens.

### Start Training

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

### Fairseq Checkpoint

We also provide the corresponding fairseq checkpoint for continual training or further modification. Coming soon.



## Citation

TBD.