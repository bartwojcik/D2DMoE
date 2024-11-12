import logging
import types

import torch
import torch.nn as nn


class AvitWrapper(nn.Module):
    def __init__(self, base_model, distilled=False, gate_scale=10.0, gate_center=30.0, eps=0.01):
        super().__init__()
        self.vit = base_model
        self.number_of_classes = self.vit.number_of_classes
        self.hidden_dim = self.vit.hidden_dim
        self.num_extra_tokens = 2 if distilled else 1
        self.grid_size = (self.vit.image_size // self.vit.patch_size, self.vit.image_size // self.vit.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.gate_scale = gate_scale  # gamma in the paper
        self.gate_center = gate_center  # beta in the paper

        def forward_act(self, x, continue_mask=None):
            bs, seq_len, _ = x.shape
            if continue_mask is None:
                y = x
                x = self.ln_1(x)
                x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
                x = self.dropout(x)
                y = x + y
                x = self.ln_2(y)
                x = self.mlp(x)
                x = x + y
            else:
                halt_mask = (~continue_mask).squeeze(-1).to(x.dtype) * -1e4
                continue_mask_converted = continue_mask.to(x.dtype)
                # see https://github.com/NVlabs/A-ViT/issues/13
                # for now add the additional mask as this seems like a more proper thing to do
                y = x
                x = self.ln_1(x) * continue_mask_converted
                x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False, key_padding_mask=halt_mask)
                x = x * continue_mask_converted
                x = self.dropout(x)
                y = x + y
                x = self.ln_2(y) * continue_mask_converted
                # x = self.mlp(x)
                x = self.mlp(x) * continue_mask_converted
                x = x + y
            token_halting_score = torch.sigmoid(x[:, :, 0] * gate_scale - gate_center)
            return x, token_halting_score.unsqueeze(-1)

        for i, l in enumerate(self.vit.encoder.layers):
            l.forward = types.MethodType(forward_act, l)
        #
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) if distilled else None
        self.pos_embed = self.vit.encoder.pos_embedding if self.num_extra_tokens == 1 else \
            nn.Parameter(torch.zeros(1, self.num_patches + self.num_extra_tokens, self.hidden_dim))
        # Representation layer
        if self.vit.representation_size and not distilled:
            self.pre_logits = self.vit.heads[:-1]
        else:
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = self.vit.heads[-1] if self.number_of_classes > 0 \
            else nn.Identity()
        self.distillation_head = None if not distilled else nn.Linear(self.hidden_dim,
                                                                      self.number_of_classes) if self.number_of_classes > 0 \
            else nn.Identity()
        self.eps = eps
        logging.info('Re-initializing the halting network bias')
        for layer_i in range(self.depth // 2):
            torch.nn.init.constant_(self.vit.encoder.layers[layer_i].mlp[-2].bias.data[0],
                                    -1. / (layer_i + 1) * gate_center)
        for layer_i in range(self.depth // 2, self.depth):
            torch.nn.init.constant_(self.vit.encoder.layers[layer_i].mlp[-2].bias.data[0],
                                    10. / (layer_i + 1) * gate_center)
        self.num_total_tokens = self.num_patches + self.num_extra_tokens

    @property
    def depth(self):
        return len(self.vit.encoder.layers)

    def forward_features(self, x):
        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(x.size(0), -1, -1)
        if self.dist_token is None:
            x = torch.cat((batch_class_token, x), dim=1)
        else:
            x = torch.cat((batch_class_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        x = self.vit.encoder.dropout(x + self.pos_embed)
        bs = x.size(0)
        token_cumul_halting_scores = torch.zeros(bs, self.num_total_tokens, 1, device=x.device)
        token_remainder_values = torch.ones(bs, self.num_total_tokens, 1, device=x.device)
        token_rho = torch.zeros(bs, self.num_total_tokens, 1, device=x.device)
        token_continue_mask = torch.ones(bs, self.num_total_tokens, 1, dtype=torch.bool, device=x.device)
        token_counter = torch.ones(bs, self.num_total_tokens, 1, dtype=torch.long, device=x.device)
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = torch.zeros_like(x)
        halting_score_means = []
        for block_i, l in enumerate(self.vit.encoder.layers):
            # block x all the parts that are not used (line 8 of Algorithm 1)
            # https://github.com/NVlabs/A-ViT/blob/master/timm/models/act_vision_transformer.py#L458
            # x.data = x.data * token_continue_mask.to(x.dtype)
            x = x * token_continue_mask.to(x.dtype)
            # evaluate layer and get halting probability for each sample
            # (line 8 of Algorithm 1)
            x, token_halting_scores = l(x, token_continue_mask)
            # token_halting_scores[1:] omits the first sample/image
            # should the class token be dropped here instead?
            # I am not first to notice this: https://github.com/NVlabs/A-ViT/issues/4 - unanswered by the authors
            # TODO change accordingly when authors address this
            # for now do not drop the class token to be closer to the actual original implementation
            # and makes more sense - class token should not be dropped early (due to ponder loss only)
            mean_halting_score = torch.mean(token_halting_scores)
            # mean_halting_score = torch.mean(token_halting_scores[:, 1:])
            halting_score_means.append(mean_halting_score)
            # Is this the last layer in the block? (line 11 of Algorithm 1)
            if block_i == len(self.vit.encoder.layers) - 1:
                token_halting_scores = torch.ones(bs, self.num_total_tokens, 1, device=x.device)
            # (lines 14-15 of Algorithm 1)
            token_cumul_halting_scores = token_cumul_halting_scores + token_halting_scores
            token_rho = token_rho + token_continue_mask.to(x.dtype)
            # Case 1: threshold reached in this iteration
            # token part (lines 19-20 of Algorithm 1)
            token_reached_mask = (token_cumul_halting_scores > 1 - self.eps).to(x.dtype) \
                                 * token_continue_mask.to(x.dtype)
            token_rho = token_rho + token_remainder_values * token_reached_mask
            # (lines 25-26 of Algorithm 1)
            delta1 = x * token_remainder_values * token_reached_mask
            # Case 2: threshold not reached
            # token part (lines 17-18 of Algorithm 1)
            tokens_not_reached_mask = (token_cumul_halting_scores < 1 - self.eps).to(x.dtype)
            token_remainder_values = token_remainder_values - (tokens_not_reached_mask * token_halting_scores)
            # (lines 23-24 of Algorithm 1)
            delta2 = x * token_halting_scores * tokens_not_reached_mask
            # counts the number of executed layers for each token
            token_counter = token_counter + tokens_not_reached_mask.to(
                torch.long)  # These data points will need at least one more layer
            # Update the mask (line 28 of Algorithm 1)
            token_continue_mask = token_cumul_halting_scores < 1 - self.eps
            # update output
            output = output + (delta1 + delta2)
        # TODO perhaps ask the authors:
        # lines 23-27 of Algorithm 1 weigh only the class token
        # while in the code authors weigh all the tokens
        # this looks like a minor discrepancy vs what is described in the paper
        # it affects the training process
        x = self.vit.encoder.ln(output)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), token_rho, token_counter, halting_score_means
        else:
            return (x[:, 0], x[:, 1]), token_rho, token_counter, halting_score_means

    def forward(self, x, return_counts=False):
        x, token_rho, token_counter, halting_scores = self.forward_features(x)
        if self.distillation_head is not None:
            x, x_dist = self.head(x[0]), self.distillation_head(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                x = (x, x_dist)
            else:
                # during inference, return the average of both classifier predictions
                x = (x + x_dist) / 2
        else:
            x = self.head(x)
        if self.training and not torch.jit.is_scripting():
            return x, token_rho, token_counter.detach(), halting_scores
        elif return_counts:
            return x, token_counter.detach()
        else:
            return x
