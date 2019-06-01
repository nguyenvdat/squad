"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
    Dat Nguyen
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import QANet, TransformerXL, BertMd
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
import os


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # setup_args = get_setup_args()
    with open(args.char2idx_file, "r") as f:
        char2idx = json_load(f)
    with open(args.idx2word_file, "r") as f:
        idx2word = json_load(f)
    # Get model
    log.info('Building model...')
    # model = QANet(word_vectors=word_vectors, char2idx = char2idx)
    model = BertMd(idx2word)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    # optimizer = optim.Adam(model.parameters())
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    outfile = '/content/dataset/train_light.npz'
    # outfile2 = '/content/dataset/train_light2.npz'
    dataset1 = np.load(args.train_record_file)

    # train_dataset = SQuAD(outfile, args.use_squad_v2)
    # # train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    # train_loader = data.DataLoader(train_dataset,
    #                                batch_size=args.batch_size,
    #                                shuffle=True,
    #                                num_workers=args.num_workers,
    #                                collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    train_dataset_len = len(dataset1['context_idxs'])
    # train_epoch_len = train_dataset_len // 2
    train_epoch_len = 20000
    print('train_dataset_len: ' + str(train_dataset_len))
    print('train_epoch_len: ' + str(train_epoch_len))
    epoch = step // train_epoch_len
    idx = np.random.choice(train_dataset_len, train_epoch_len, replace=False)
    while epoch != args.num_epochs:
        idx = np.random.choice(
            train_dataset_len, train_epoch_len, replace=False)
        # if epoch % 2 == 0:
        #     idx = np.random.choice(
        #         train_dataset_len, train_epoch_len, replace=False)
        # else:
        #     idx2 = list(set(range(train_dataset_len)) - set(idx))
        #     idx = idx2
        train_dataset = None
        train_loader = None
        np.savez(outfile, context_idxs=dataset1['context_idxs'][idx],
                 context_char_idxs=dataset1['context_char_idxs'][idx],
                 ques_idxs=dataset1['ques_idxs'][idx],
                 ques_char_idxs=dataset1['ques_char_idxs'][idx],
                 y1s=dataset1['y1s'][idx],
                 y2s=dataset1['y2s'][idx],
                 ids=dataset1['ids'][idx])
        train_dataset = SQuAD(outfile, args.use_squad_v2)
        # train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       collate_fn=collate_fn)

        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
                # tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in dev_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p1, log_p2, y1, y2 = model(cw_idxs, qw_idxs, y1, y2)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()
                # print('loss val: {}'.format(loss_val))

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step(step // batch_size)
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    # if True:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2, y1, y2, y_maps = model(cw_idxs, qw_idxs, y1, y2)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
            # print('Starts: {}'.format(starts))
            # print('Ends : {}'.format(ends))
            # print('y1: {}'.format(y1))
            # print('y2: {}'.format(y2))

            # Log info
            # progress_bar.update(batch_size)
            # progress_bar.set_postfix(NLL=nll_meter.avg)

            starts = y_maps[range(len(y_maps)), starts.tolist()]
            ends = y_maps[range(len(y_maps)), ends.tolist()]
            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts,
                                           ends,
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
