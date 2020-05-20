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


def make_infinite_list(personas):
    while True:
        print("New epoch")
        shuffle(personas)
        for x in personas:
            yield x


def do_learning_fix_step(model, train_iter, val_iter, iterations):
    model.train()
    val_ppl = []
    val_loss = 0
    for _ in range(iterations):
        for data in train_iter:
            model.train(data)
    for data in val_iter:
        _, ppl, loss_tensor = model(data)
        val_loss += loss_tensor
        val_ppl.append(ppl)
    return val_loss / len(val_ppl), np.mean(val_ppl)


def do_evaluation(model, test_iter):
    model.eval()
    with torch.no_grad():
        ppl_list, loss_list = [], []
        for batch in test_iter:
            loss, ppl, _ = model(batch)
            loss_list.append(loss)
            ppl_list.append(ppl)
    return np.mean(loss_list), np.mean(ppl_list)


p = Personas()
# Make save_path
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
optimizer_map = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}
meta_optimizer = optimizer_map[config.meta_optimizer](
    meta_net.parameters(), lr=config.meta_lr)

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
        train_iter, val_iter = p.get_loader(
            persona=tasks_iter.__next__(),
            batch_size=config.batch_size,
            split='train',
            balanced=config.fix_dialnum_train,
        )
        # before first update
        val_loss_before, val_ppl_before = do_evaluation(meta_net, val_iter)
        train_loss_before.append(val_ppl_before)
        # Update fast nets
        val_loss_update, val_ppl_update = do_learning_fix_step(
            meta_net, train_iter, val_iter, iterations=config.meta_iteration)
        val_loss_update_from_eval, val_ppl_update_from_eval = \
            do_evaluation(meta_net, val_iter)
        print(
            f"meta_iteration {meta_iteration} "
            f"meta_batch_index {meta_batch_index}: "
            f"val_loss_before {val_loss_before} val_ppl_before {val_ppl_before}"
            f"val_loss_update {val_loss_update} val_ppl_update {val_ppl_update}"
            f"val_loss_update_from_eval {val_loss_update_from_eval}"
            f"val_ppl_update_from_eval {val_ppl_update_from_eval}"
        )
        train_loss_meta.append(val_ppl_update)
        batch_loss += val_loss_update
        # reset
        meta_net.load_state_dict(
            {name: weights_original[name] for name in weights_original})

    writer.add_scalars('loss_before',
                       {'train_loss_before': np.mean(train_loss_before)},
                       meta_iteration)
    writer.add_scalars('loss_meta',
                       {'train_loss_meta': np.mean(train_loss_meta)},
                       meta_iteration)
    print(
        f"train_loss_before: {np.mean(train_loss_before)} "
        f"+- {np.std(train_loss_before)}")
    print(
        f"train_loss_meta: {np.mean(train_loss_meta)} "
        f"+- {np.std(train_loss_meta)}")

    # meta Update
    meta_optimizer.zero_grad()
    batch_loss /= meta_batch_size
    batch_loss.backward()
    # clip gradient
    nn.utils.clip_grad_norm_(meta_net.parameters(), config.max_grad_norm)
    meta_optimizer.step()

    # Meta-Evaluation
    if meta_iteration % 10 == 0 and meta_iteration:
        print('Meta_iteration:', meta_iteration)
        val_loss_before = []
        val_loss_meta = []
        weights_original = deepcopy(meta_net.state_dict())
        for per in p.get_personas('valid'):
            train_iter, val_iter = p.get_loader(
                persona=per,
                batch_size=config.batch_size,
                split='valid',
                fold=0,
                balanced=config.fix_dialnum_train,
            )
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
