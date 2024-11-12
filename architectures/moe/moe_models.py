from typing import Dict

import torch
from torch import nn
from transformers import apply_chunking_to_forward
from transformers.models.bert import BertLayer
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.custom import CustomMultiheadAttention
from architectures.moe.moe_layers import MoELayer


def moe_attention_forward(self: CustomMultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          _key_padding_mask=None,
                          need_weights=False):
    # TODO warning! works only with "simplified" MultiheadAttention
    assert query.size() == key.size() == value.size()
    batch_size, seq_length, embed_dim = query.size()
    gating_data = {}
    if isinstance(self.q_proj, MoELayer):
        query, q_gating_data = self.q_proj(query)
        gating_data.update(q_gating_data)
    else:
        query = self.q_proj(query)
    if isinstance(self.k_proj, MoELayer):
        key, k_gating_data = self.k_proj(key)
        gating_data.update(k_gating_data)
    else:
        key = self.k_proj(key)
    if isinstance(self.v_proj, MoELayer):
        value, v_gating_data = self.v_proj(value)
        gating_data.update(v_gating_data)
    else:
        value = self.v_proj(value)
    # separate the head dimension, and permute dimensions into [Batch, Head, SeqLen, Dims]
    query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    # Determine value outputs
    values, attention = self.scaled_dot_product(query, key, value,
                                                dropout_p=self.dropout_p if self.training else 0.0)
    values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
    values = values.reshape(batch_size, seq_length, embed_dim)
    if isinstance(self.o_proj, MoELayer):
        o, o_routing_data = self.o_proj(values)
        gating_data.update(o_routing_data)
    else:
        o = self.o_proj(values)
    if need_weights:
        return o, attention, gating_data
    else:
        return o, None, gating_data


def moe_vit_block_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    # torch.isnan is a synchronizing cuda operation - disable for benchmarking
    # assert not torch.any(torch.isnan(input)), f'{input=}'
    gating_data = {}
    x = self.ln_1(input)
    # some MHA layers can have MoEs instead of projection matrices
    if isinstance(self.self_attention, CustomMultiheadAttention):
        x, _, attn_gating_data = self.self_attention(query=x, key=x, value=x, need_weights=False)
        gating_data.update(attn_gating_data)
    else:
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
    x = self.dropout(x)
    x = x + input
    y = self.ln_2(x)
    # not all FFN layers have to be replaced with a MoE layer
    if isinstance(self.mlp, MoELayer):
        y, ffn_gating_data = self.mlp(y)
        gating_data.update(ffn_gating_data)
    else:
        y = self.mlp(y)
    return x + y, gating_data


def moe_gpt_block_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    assert not torch.any(torch.isnan(input)), f'{input=}'
    gating_data = {}
    ln1 = self.ln_1(input)
    if isinstance(self.attn, CustomMultiheadAttention):
        attn_output, _, attn_gating_data = self.attn(query=ln1, key=ln1, value=ln1, need_weights=False)
        # TODO how to preserve dropout there?
        gating_data.update(attn_gating_data)
    else:
        attn_output = self.attn(ln1)
    h = input + attn_output
    ln2 = self.ln_2(h)
    # not all FFN layers have to be replaced with a MoE layer
    if isinstance(self.mlp, MoELayer):
        mlp_out, ffn_gating_data = self.mlp(ln2)
        # TODO is there a better way to do dropout here?
        mlp_out = self.mlp.dropout(mlp_out)
        gating_data.update(ffn_gating_data)
    else:
        mlp_out = self.mlp(ln2)
    return h + mlp_out, gating_data


def moe_vit_encoder_forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    encoder_block_x = self.dropout(input + self.pos_embedding)
    gating_data = {}
    for layer in self.layers:
        encoder_block_x, encoder_block_gating_data = layer(encoder_block_x)
        gating_data.update(encoder_block_gating_data)
    return self.ln(encoder_block_x), gating_data


def moe_vit_main_forward(self, x: torch.Tensor, return_gating_data: bool = False):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]
    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x, gating_data = self.encoder(x)
    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    x = self.heads(x)
    if return_gating_data is True:
        return x, gating_data
    else:
        return x


def moe_gemma_decoder_forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                              position_ids: torch.Tensor, cache_position: torch.Tensor, ):
    gating_data = {}

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    if isinstance(self.mlp, MoELayer):
        hidden_states, ffn_gating_data = self.mlp(hidden_states)
        gating_data.update(ffn_gating_data)
    else:
        hidden_states = self.mlp(hidden_states)

    hidden_states = residual + hidden_states

    return hidden_states, gating_data


def moe_gemma_main_forward(self, input_ids: torch.Tensor, attention_mask, return_gating_data: bool = False):
    inputs_embeds = self.model.embed_tokens(input_ids)
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = self.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, None, False)
    hidden_states = inputs_embeds
    # normalized
    # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer
    gating_data = {}
    for decoder_layer in self.model.layers:
        layer_outputs, transformer_block_gating_data = decoder_layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        gating_data.update(transformer_block_gating_data)
        hidden_states = layer_outputs
    hidden_states = self.model.norm(hidden_states)
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    if return_gating_data is True:
        return logits, gating_data
    else:
        return logits


def moe_gpt_main_forward(self, x: torch.Tensor, return_gating_data: bool = False):
    # Reshape and permute the input tensor
    device = x.device
    b, t = x.size()
    assert (
            t <= self.config.block_size
    ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    tok_emb = self.transformer.wte(x)  # token embeddings of shape (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

    x = self.transformer.drop(tok_emb + pos_emb)
    gating_data = {}
    for block in self.transformer.h:
        x, transformer_block_gating_data = block(x)
        gating_data.update(transformer_block_gating_data)

    x = self.transformer.ln_f(x)
    x = self.lm_head(x)

    if return_gating_data is True:
        return x, gating_data
    else:
        return x


def moe_bert_layer_forward(self, x: torch.Tensor, attn_mask: torch.Tensor, return_gating_data: bool = False):
    if isinstance(self.attention, CustomMultiheadAttention):
        raise NotImplementedError()
    else:
        attn_output = self.attention(
            x,
            attn_mask,
        )[0]

    gating_data = {}
    if hasattr(self, 'mlp'):
        ffn_out, ffn_gating_data = self.mlp(attn_output)
        gating_data.update(ffn_gating_data)
        ffn_out = self.dropout(ffn_out)
        y = self.ln(ffn_out + attn_output)
    else:
        y = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attn_output,
        )

    return y, gating_data


def moe_bert_main_forward(self, x: Dict[str, torch.Tensor], return_gating_data: bool = False):
    input_ids = x["input_ids"]
    attention_mask = x["attention_mask"]
    token_type_ids = x["token_type_ids"]

    # BEGIN ENCODER
    # should be equivalent to: x = self.encoder(x)
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        attention_mask, input_shape=input_ids.size()
    )
    x = self.bert.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
    )

    # Iterate through the transformer blocks
    gating_data = {}
    for bert_block in self.bert.encoder.layer:
        x, block_gating_data = bert_block(x, extended_attention_mask)
        gating_data.update(block_gating_data)

    # Return classifier token
    pooled_output = self.bert.pooler(x) if self.bert.pooler is not None else x
    pooled_output = self.dropout(pooled_output)
    x = self.classifier(pooled_output)

    if return_gating_data is True:
        return x, gating_data
    else:
        return x


def ffn_filter_condition_bert(_model: nn.Module, m: nn.Module):
    if isinstance(m, BertLayer):
        return True


def ffn_filter_condition_gemma(_model: nn.Module, m: nn.Module):
    if isinstance(m, GemmaMLP):
        return True
