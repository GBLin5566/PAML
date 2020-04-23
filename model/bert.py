import os
import math

import torch
import torch.nn as nn
from model import HuggingfaceEncoderDecoder
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

        self.model = HuggingfaceEncoderDecoder.from_pretrained(
            'bert-base-uncased',
            'bert-base-uncased',
        )

        if is_eval:
            self.model = self.model.eval()

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_idx)
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

    def train_one_batch(self, batch, train=True):
        if train:
            self.model.train()
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = \
            get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        dec_batch_input, dec_batch_output = dec_batch[:, 1:], dec_batch[:, :-1]

        self.optimizer.zero_grad()
        logit, *_ = self.model(enc_batch, dec_batch_input)

        # loss: NNL if ptr else Cross entropy
        loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch_output.contiguous().view(-1))

        if(config.act):
            loss += self.compute_act_loss(self.encoder)
            loss += self.compute_act_loss(self.decoder)

        if(train):
            loss.backward()
            self.optimizer.step()
        if(config.label_smoothing):
            loss = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1))

        return loss.item(), math.exp(min(loss.item(), 100)), loss
