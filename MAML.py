import os
import math
from copy import deepcopy
from random import shuffle

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from model import Bert2Bert, Bart
from utils import config
from utils.data_reader import Personas


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def make_infinite_list(personas):
    while True:
        print("New epoch")
        shuffle(personas)
        for x in personas:
            yield x


def do_learning_early_stop(model, train_iter, val_iter, iterations, strict=1):
    # b_loss, b_ppl = do_evaluation(model, val_iter)
    b_loss, b_ppl = 100000, 100000
    best = deepcopy(model.state_dict())
    cnt = 0
    idx = 0
    for _ in range(iterations):
        train_l, train_p = [], []
        for d in train_iter:
            t_loss, t_ppl, _ = model.train_one_batch(d)
            train_l.append(t_loss)
            train_p.append(t_ppl)

        n_loss, n_ppl = do_evaluation(model, val_iter)
        # early stopping
        if(n_ppl <= b_ppl):
            b_ppl = n_ppl
            b_loss = n_loss
            cnt = 0
            idx += 1
            best = deepcopy(model.state_dict())  # save best weights
        else:
            cnt += 1
        if(cnt > strict):
            break

    # load the best model
    model.load_state_dict({name: best[name] for name in best})

    return (np.mean(train_l), np.mean(train_p), b_loss, b_ppl), idx


def do_learning_fix_step(model, train_iter, val_iter, iterations, test=False):
    val_p = []
    val_p_list = []
    val_loss = 0
    for _, _ in enumerate(range(iterations)):

        for d in train_iter:
            t_loss, t_ppl, _ = model.train_one_batch(d)
        if test:
            with torch.no_grad():
                _, test_ppl = do_evaluation(model, val_iter)
                val_p_list.append(test_ppl)

    if test:
        return val_p_list
    else:
        for d in val_iter:
            _, t_ppl, t_loss = model.train_one_batch(d, train=False)
            val_loss += t_loss
            val_p.append(t_ppl)
        return val_loss, np.mean(val_p)


def do_evaluation(model, test_iter):
    with torch.no_grad():
        p, l = [], []
        for batch in test_iter:
            loss, ppl, _ = model.train_one_batch(batch, train=False)
            l.append(loss)
            p.append(ppl)
    return np.mean(l), np.mean(p)


p = Personas()
path_split = config.save_path.split(os.sep)
if not path_split[-1]:
    path_split.pop(-1)
path_split[-1] += \
    f"_model_{config.model_type}_lr_{config.lr}_meta_lr_{config.meta_lr}_warmup_{config.warmup}"
save_path = f'{os.sep}'.join(path_split)
writer = SummaryWriter(log_dir=save_path)
# Build model, optimizer, and set states
build_model_func_map = {
    'bert2bert': Bert2Bert,
    # 'gpt2gpt': GPT2GPT,
    'bart': Bart,
}
meta_net = build_model_func_map[config.model_type]()
if config.meta_optimizer == 'sgd':
    meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=config.meta_lr)
elif config.meta_optimizer == 'adam':
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=config.meta_lr)
else:
    raise ValueError

meta_batch_size = config.meta_batch_size
tasks = p.get_personas('train')
tasks_iter = make_infinite_list(tasks)

# meta early stop
patience = 50
if config.fix_dialnum_train:
    patience = 100
best_before_loss = best_meta_loss = 10000000
stop_count = 0
# Main loop
for meta_iteration in range(config.epochs):
    # save original weights to make the update
    # NOTE theta = weights_original
    weights_original = deepcopy(meta_net.state_dict())
    train_loss_before = []
    train_loss_meta = []
    # loss accumulate from a batch of tasks
    batch_loss = 0
    for meta_batch_index in range(meta_batch_size):
        # Get task
        if config.fix_dialnum_train:
            train_iter, val_iter = p.get_balanced_loader(
                persona=tasks_iter.__next__(),
                batch_size=config.batch_size, split='train')
        else:
            train_iter, val_iter = p.get_data_loader(
                persona=tasks_iter.__next__(),
                batch_size=config.batch_size, split='train')
        # before first update
        v_loss, v_ppl = do_evaluation(meta_net, val_iter)
        train_loss_before.append(math.exp(v_loss))
        # Update fast nets
        val_loss, val_ppl = do_learning_fix_step(
            meta_net, train_iter, val_iter, iterations=config.meta_iteration)
        print(
            f"meta_iteration {meta_iteration} meta_batch_index {meta_batch_index}:"
            f" loss {val_loss} ppl {val_ppl}"
        )
        train_loss_meta.append(math.exp(val_loss.item()))
        batch_loss += val_loss
        # log

        # reset
        meta_net.load_state_dict(
            {name: weights_original[name] for name in weights_original})

    writer.add_scalars('loss_before',
                       {'train_loss_before': np.mean(train_loss_before)},
                       meta_iteration)
    writer.add_scalars('loss_meta',
                       {'train_loss_meta': np.mean(train_loss_meta)},
                       meta_iteration)
    print(f"train_loss_before: {np.mean(train_loss_before)} +- {np.std(train_loss_before)}")
    print(f"train_loss_meta: {np.mean(train_loss_meta)} +- {np.std(train_loss_meta)}")

    # meta Update
    if(config.meta_optimizer == 'noam'):
        meta_optimizer.optimizer.zero_grad()
    else:
        meta_optimizer.zero_grad()
    batch_loss /= meta_batch_size
    batch_loss.backward()
    # clip gradient
    nn.utils.clip_grad_norm_(meta_net.parameters(), config.max_grad_norm)
    meta_optimizer.step()
    batch_loss.detach()

    # Meta-Evaluation
    if meta_iteration % 10 == 0 and meta_iteration:
        print('Meta_iteration:', meta_iteration)
        val_loss_before = []
        val_loss_meta = []
        weights_original = deepcopy(meta_net.state_dict())
        for per in p.get_personas('valid'):
            if config.fix_dialnum_train:
                train_iter, val_iter = p.get_balanced_loader(
                    persona=per,
                    batch_size=config.batch_size, split='valid', fold=0)
            else:
                train_iter, val_iter = p.get_data_loader(
                    persona=per,
                    batch_size=config.batch_size, split='valid', fold=0)
            # zero shot result
            loss, ppl = do_evaluation(meta_net, val_iter)
            val_loss_before.append(math.exp(loss))
            # mata tuning
            val_loss, val_ppl = do_learning_fix_step(
                meta_net, train_iter, val_iter,
                iterations=config.meta_iteration)
            print(f"persona {per}: loss {val_loss} ppl {val_ppl}")
            val_loss_meta.append(math.exp(val_loss.item()))
            # updated result

            meta_net.load_state_dict(
                {name: weights_original[name] for name in weights_original})

        writer.add_scalars(
            'loss_before', {
                'val_loss_before': np.mean(val_loss_before)}, meta_iteration)
        writer.add_scalars(
            'loss_meta', {
                'val_loss_meta': np.mean(val_loss_meta)}, meta_iteration)
        print(f"val_loss_before: {np.mean(val_loss_before)} +- {np.std(val_loss_before)}")
        print(f"val_loss_meta: {np.mean(val_loss_meta)} +- {np.std(val_loss_meta)}")
        # check early stop
        if np.mean(val_loss_before) < best_before_loss:
            best_before_loss = np.mean(val_loss_before)
        if np.mean(val_loss_meta) < best_meta_loss:
            best_meta_loss = np.mean(val_loss_meta)
            stop_count = 0
            meta_net.save_model(best_meta_loss, 1, 0.0, 0.0, 0.0, 1.1)
        else:
            stop_count += 1
            if stop_count > patience:
                break
        print(f"Current best_before_loss: {best_before_loss}")
        print(f"Current best_meta_loss: {best_meta_loss}")
        print()
