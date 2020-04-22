import os
import math
import random

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
            vocab,
            is_eval=False,
            ):
        super(Bert2Bert, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.model = HuggingfaceEncoderDecoder.from_pretrained(
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

    def train_one_batch(self, batch, train=True):
        # pad and other stuff
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = \
            get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        self.optimizer.zero_grad()

        # TODO
        import ipdb; ipdb.set_trace()
        output = self.model(enc_batch, dec_batch)

        # Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)

        # Decode
        sos_token = torch.LongTensor(
            [config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA:
            sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(self.embedding(
            dec_batch_shift), encoder_outputs, (mask_src, mask_trg))
        # compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab,
            extra_zeros)

        # loss: NNL if ptr else Cross entropy
        loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1))

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
