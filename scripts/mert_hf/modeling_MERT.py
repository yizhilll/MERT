"""
MERT model definition.
We largely adapt codes from:
1. https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py
2. https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py
"""

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput
import torch
from torch import nn

from transformers.models.hubert.modeling_hubert import (
    HubertFeatureEncoder,
    HubertModel,
    HubertEncoderStableLayerNorm,
    HubertEncoder,
    HubertEncoderLayer,
    HubertPositionalConvEmbedding,
    HubertAttention,
    HubertFeedForward,
)

try:
    from nnAudio import features as nnAudioFeatures
    NNAUDIO_INSTALLED=True
except:
    print("WARNING: feature_extractor_cqt requires the libray 'nnAudio'")
    NNAUDIO_INSTALLED=False

from .configuration_MERT import MERTConfig

class MERTFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        self.feature_extractor_cqt = config.feature_extractor_cqt

        if self.feature_extractor_cqt:
            # v3 concat features
            self.feature_dimension = config.conv_dim[-1] + config.feature_extractor_cqt_bins
            print(f"feature dimention: {self.feature_dimension}")
        else:
            self.feature_dimension = config.conv_dim[-1]
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(self.feature_dimension, eps=config.layer_norm_eps)
        self.projection = nn.Linear(self.feature_dimension, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MERTModel(HubertModel):
    # overwrite config class
    config_class = MERTConfig
    base_model_prefix = "mert_model"
    def __init__(
        self,
        config: MERTConfig,
    ) -> None:
        """ 
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        super(HubertModel, self).__init__(config)

        self.config = config

        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MERTFeatureProjection(config) # replace Feature Projection for introcuing new feature

        if self.config.feature_extractor_cqt:
            assert NNAUDIO_INSTALLED, "ERROR: feature_extractor_cqt requires the libray 'nnAudio', try after `pip install nnAudio` "
            print('initializing cqt extractor for MERT')            
            self.feature_extractor_cqt = nnAudioFeatures.cqt.CQT(sr=self.config.sample_rate, hop_length=self.config.sample_rate//50, fmin=32.7, 
                    fmax=None, n_bins=self.config.feature_extractor_cqt_bins, bins_per_octave=self.config.feature_extractor_cqt_bins//7, 
                    filter_scale=1, norm=1, window='hann', center=True, 
                    pad_mode='constant', trainable=False, 
                    output_format='Magnitude', verbose=True)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        
        if config.do_stable_layer_norm:
            assert not config.deepnorm, "must use post-layer_norm with deepnorm"
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            if config.deepnorm:
                self.encoder = HubertEncoder_extend(config)
            else:
                self.encoder = HubertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None, mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        
        # return super().forward(input_values, attention_mask, mask_time_indices, output_attentions, output_hidden_states, return_dict)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        # add additional cqt features for transformer input
        if self.config.feature_extractor_cqt:
            features_cqt = self.feature_extractor_cqt(input_values).transpose(1, 2)
            features_cqt = features_cqt[:,:extract_features.shape[1],:] # align shape
            # # v2
            # features_cqt = self.post_cqt_feature_proj(features_cqt)
            # extract_features = self.feature_projection.layer_norm(extract_features) + self.feature_projection.layer_norm(features_cqt) #v2
            # v3
            extract_features = torch.cat([extract_features,features_cqt], 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class HubertEncoder_extend(HubertEncoder):
    def __init__(self, config):
        # super().__init__()
        # call nn module initialization
        nn.Module.__init__(self)
        # super(HubertEncoder_extend, self).__init__()

        self.config = config
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

        
        self.layers = nn.ModuleList([HubertEncoderLayerExtend(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

        if config.deepnorm:
            import math
            init_scale = math.pow(8.0 * config.num_hidden_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "feed_forward.intermediate_dense" in name
                    or "feed_forward.output_dense" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

class HubertEncoderLayerExtend(HubertEncoderLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        # super(HubertEncoderLayerExtend, self).__init__()
        if config.attention_relax > 0 :
            self.attention = HubertAttention_extend(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=False,
                attention_relax=config.attention_relax,
            )
        else:    
            self.attention = HubertAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=False,
            )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.deepnorm:
            import math
            self.residual_alpha = math.pow(2.0 * config.num_hidden_layers, 0.25)
        else:
            self.residual_alpha = 1.0
    
    def residual_connection(self, x, residual):
        '''
        residual: input before f()
        x: output of f(residual)
        '''
        return residual * self.residual_alpha + x

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)

        # hidden_states = attn_residual + hidden_states
        hidden_states = self.residual_connection(hidden_states, attn_residual)

        hidden_states = self.layer_norm(hidden_states)

        # hidden_states = hidden_states + self.feed_forward(hidden_states)
        ffn_residual = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.residual_connection(hidden_states, ffn_residual)

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class HubertAttention_extend(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        attention_relax: float = -1.0,
    ):
        super().__init__()
        # nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if attention_relax > 0:
            self.attention_relax = attention_relax

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if self.attention_relax > 0:
            # => (bsz, self.num_heads, tgt_len, src_len)
            # attn_weights_relax = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)/self.attention_relax
            # => (bsz*self.num_heads, tgt_len, src_len)
            attn_weights_relax = attn_weights / self.attention_relax

            # => (bsz* self.num_heads, tgt_len, 1)
            attn_max_relax = torch.max(attn_weights_relax, dim=-1, keepdim=False).unsqueeze(2)
            attn_weights = (attn_weights_relax - attn_max_relax) * self.attention_relax

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value