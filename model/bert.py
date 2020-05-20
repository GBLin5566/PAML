import os
import math

import torch
import torch.nn as nn
from transformers import EncoderDecoderModel

from model.common_layer import (
    get_input_from_batch,
    get_output_from_batch,
)
from utils import config


class Bert2Bert(nn.Module):

    def __init__(
            self,
            is_eval=False,
            ):
        super().__init__()

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            'bert-base-uncased',
            'bert-base-uncased',
        )

        if is_eval:
            self.model = self.model.eval()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        if config.use_sgd:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)

        if config.USE_CUDA:
            self.model = self.model.cuda()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        state = {
            'iter': iter,
            'model_state_dict': self.model.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(
            self.model_dir,
            'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(
                    iter, running_avg_ppl, f1_g, f1_b, ent_g, ent_b))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def forward(self, batch):
        enc_batch, enc_mask, _, enc_batch_extend_vocab, extra_zeros, _, _ = \
            get_input_from_batch(batch)
        dec_batch, dec_mask, _, _, _ = get_output_from_batch(batch)
        dec_batch_input, dec_batch_output = dec_batch[:, :-1], dec_batch[:, 1:]
        dec_mask = dec_mask[:, :-1]

        self.optimizer.zero_grad()
        loss = self.model(
            input_ids=enc_batch,
            decoder_input_ids=dec_batch_input,
            lm_labels=dec_batch_output,
            attention_mask=enc_mask,
            decoder_attention_mask=dec_mask,
        )[0]

        return loss.item(), math.exp(min(loss.item(), 600)), loss

    def train(self, batch):
        loss_value, ppl, loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        return loss_value, ppl
