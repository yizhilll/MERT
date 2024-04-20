# coding=utf-8

"""Convert Hubert & MERT checkpoint."""

import argparse
import json

from pydantic import conint

import fairseq
import torch
from fairseq.data import Dictionary

from transformers import (
    HubertConfig,
    HubertForCTC,
    HubertModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)
from datasets import load_dataset

import numpy as np


from ..mert_hf.configuration_MERT import MERTConfig
from ..mert_hf.modeling_MERT import MERTModel

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}

SKIP_MODULE = ['encoder_cqt_model', 'encoder_mel_model']

def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.hubert.feature_extractor if is_finetuned else hf_model.feature_extractor

    for name, value in fairseq_dict.items():
        # if name in SKIP_MODULE:
        if "encoder_cqt_model" in name \
            or 'encoder_mel_model' in name :
            logger.warning(f'skip module in fairseq checkpoint in the skip list: {name}')
            continue
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                mapped_key = "hubert." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                if key in name or (key.split("w2v_model.")[-1] == name.split(".")[0] and not is_finetuned):
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "bias" in name:
                        weight_type = "bias"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)

@torch.no_grad()
def verify_conversion(model, hf_wav2vec, config, is_finetuned):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60", sampling_rate=config.sample_rate)

    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    # input_audio = [x["array"] for x in ds[:4]["audio"]]
    
    input_audio = [np.random.randn(74400)]
    input_audio.append(np.random.randn(104560))
    input_audio.append(np.random.randn(213360))
    input_audio.append(np.random.randn(86720))

    inputs = processor(input_audio, return_tensors="pt", padding=True, sampling_rate=config.sample_rate)

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask
    #    input_values = inputs.input_values[:, :-1]
    #    attention_mask = inputs.attention_mask[:, :-1]

    hf_wav2vec.eval()
    model.eval()
    if is_finetuned:
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)[
            "encoder_out"
        ].transpose(0, 1)
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["logits"]

        pred_ids = torch.argmax(our_output, dim=-1)
        output_string = processor.batch_decode(pred_ids)

        # print(f"Expected Output: {ds[:4]['text']}, Pred: {output_string}")
    else:
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)["layer_results"]
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask, output_hidden_states=True)["hidden_states"]
        
        logger.info(f'shape of all their_output: {len(their_output), their_output[0][0].shape}; shape of all our_output: {len(our_output), our_output[0].shape}')

        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)["x"]
        logger.info(f'from their_outputs[x][0,:5,:5] {their_output[0,:5,:5]}')
                    
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)[
            "layer_results"
        ][-1][0].transpose(0, 1)
        # their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True, output_layer=None)[
        #     "x"
        # ]
        # logger.info(f'hf attention_mask: {attention_mask}')
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask, output_hidden_states=True)["last_hidden_state"]
        # our_output = hf_wav2vec(input_values, attention_mask=attention_mask, output_hidden_states=True)["hidden_states"][0]
        logger.info(f'sample of their_output[0,:5,:5]: {their_output[0,:5,:5]}; sample of our_output[0,:5,:5]: {our_output[0,:5,:5]}')

    logger.info(f'shape of their_output: {their_output.shape}; shape of our_output: {our_output.shape}')
    # print(our_output.shape, their_output.shape)

    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")


@torch.no_grad()
def convert_hubert_checkpoint(
    args, checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = MERTConfig.from_pretrained(config_path)
    else:
        config = MERTConfig()

    if config.deepnorm:
        logger.info('WARNING: initialize the model with DeepNet transformer layer')

    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # important change bos & pad token id since CTC symbol is <pad> and
            # not <s> as in fairseq
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=config.sample_rate,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)

        hf_wav2vec = HubertForCTC(config)
    else:
        hf_wav2vec = MERTModel(config)

    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        # model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path],)
            # arg_overrides={ 
            #     'model': {'_name':"hubert"},
            #     'task' : {'_name': 'hubert_pretraining'}
            # }) # , "common":{"user_dir":args.user_dir}, "task": {"_name": "audio_pretraining"}

    model = model[0].eval()

    recursively_load_weights(model, hf_wav2vec, is_finetuned)

    # verify_conversion(model=model, hf_wav2vec=hf_wav2vec, config=config, is_finetuned=is_finetuned)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    print(__package__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    parser.add_argument("--user_dir", default=None, type=str, help="Path to custom models")
    args = parser.parse_args()
    convert_hubert_checkpoint(
        args, args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
    
    