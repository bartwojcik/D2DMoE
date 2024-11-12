import os
from functools import partial
from pathlib import Path

import torch
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    GemmaConfig,
    GemmaForCausalLM,
    apply_chunking_to_forward,
)

from architectures.gpt import GPT, GPTConfig


def get_bert(model_name_or_path, num_classes, max_seq_length):
    model = BertForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_classes,
        cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"],
    )

    def forward_generator(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        token_type_ids = x["token_type_ids"]

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape=input_ids.size()
        )
        x = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        # go through encoder blocks
        for block in self.bert.encoder.layer:
            x = block.attention(
                x,
                extended_attention_mask,
            )[0]
            x = yield x, None

            x = apply_chunking_to_forward(
                block.feed_forward_chunk,
                block.chunk_size_feed_forward,
                block.seq_len_dim,
                x,
            )

            x = yield x, None

        # END OF ENCODER
        # classifier token
        pooled_output = self.bert.pooler(x) if self.bert.pooler is not None else x
        pooled_output = self.dropout(pooled_output)
        x = self.classifier(pooled_output)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        token_type_ids = x["token_type_ids"]
        return org_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        "input_ids": max_seq_length,
        "token_type_ids": max_seq_length,
        "attention_mask": max_seq_length,
    }
    model.number_of_classes = num_classes

    return model


def get_distilbert(model_name_or_path, num_classes, max_seq_length):
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_classes,
        cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"],
    )

    def forward_generator(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = self.distilbert.embeddings(
            input_ids=input_ids,
        )
        # go through encoder blocks
        for block in self.distilbert.transformer.layer:
            x = block(
                x,
                attention_mask,
            )[-1]

            x = yield x, None

        # END OF ENCODER
        # classifier token
        x = x[:, 0]  # (bs, dim)
        x = self.pre_classifier(x)  # (bs, dim)
        x = torch.nn.ReLU()(x)  # (bs, dim)
        x = self.dropout(x)  # (bs, dim)
        x = self.classifier(x)  # (bs, num_labels)

        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        return org_forward(input_ids=input_ids, attention_mask=attention_mask).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        "input_ids": max_seq_length,
        "attention_mask": max_seq_length,
    }
    model.number_of_classes = num_classes

    return model


def get_roberta(model_name_or_path, num_classes, max_seq_length):
    model = RobertaForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_classes,
        cache_dir=os.environ["TRANSFORMERS_CACHE_DIR"],
    )

    def forward_generator(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape=input_ids.size()
        )
        x = self.roberta.embeddings(
            input_ids=input_ids,
        )
        # go through encoder blocks
        for block in self.roberta.encoder.layer:
            x = block(
                x,
                attention_mask=extended_attention_mask,
            )[0]

            x = yield x, None

        # END OF ENCODER
        # classifier token
        pooled_output = self.roberta.pooler(x) if self.roberta.pooler is not None else x
        x = self.classifier(pooled_output)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        return org_forward(input_ids=input_ids, attention_mask=attention_mask).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        "input_ids": max_seq_length,
        "attention_mask": max_seq_length,
    }
    model.number_of_classes = num_classes

    return model


def get_gpt2(
        model_name=None,
        block_size=None,
        n_layer=None,
        n_head=None,
        n_embd=None,
        bias=None,
        dropout=0.0,
        meta_vocab_size=None,
        activation=None,
):
    if model_name is not None:
        model = GPT.from_pretrained(model_name, override_args={"dropout": dropout})
    else:
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
            dropout=dropout,
            activation=activation,
        )
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model.config.block_size = block_size  # so that the checkpoint will have the right value

    def forward_generator(self, x):
        device = x.device
        b, t = x.size()
        assert (
                t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(x)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = x + block.attn(block.ln_1(x))
            x = yield x, None
            x = x + block.mlp(block.ln_2(x))
            x = yield x, None

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    # TODO check what has to go here
    model.input_size = 256
    model.input_channels = 3
    model.number_of_classes = model.config.vocab_size
    return model


def get_gemma_2b(cache_dir=None):
    return get_gemma('google/gemma-2b', cache_dir=cache_dir)


class GemmaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gemma = model

    def forward(self, x, return_gating_data=False):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        if return_gating_data:
            return self.gemma(input_ids=input_ids, attention_mask=attention_mask, return_gating_data=True)
        else:
            output = self.gemma(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(output, 'logits'):
            # Unwrap huggingface outputs if using standard model without wrapped forward
            return output.logits
        else:
            return output


def get_gemma(
        model_name=None,
        num_hidden_layers=None,
        hidden_size=None,
        intermediate_size=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        cache_dir=None
):
    if cache_dir is None:
        cache_dir = os.environ["TRANSFORMERS_CACHE_DIR"]
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if model_name is not None:
        model = GemmaForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
    else:
        config = GemmaConfig.from_pretrained('google/gemma-2b')
        config.num_hidden_layers = num_hidden_layers
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads
        model = GemmaForCausalLM(config)

    model = GemmaWrapper(model)
    # TODO check what has to go here
    model.input_channels = 3
    model.number_of_classes = model.gemma.config.vocab_size
    return model
