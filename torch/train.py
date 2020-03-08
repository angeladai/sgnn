from __future__ import division
from __future__ import print_function

import argparse
import os, sys, time
import shutil
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
import model
import loss as loss_util


# python train.py --gpu 0 --data_path ./data/completion_blocks_2cm_hierarchy/release_64-64-128  --train_file_list train_list.txt --val_file_list val_list.txt --save logs/mp

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data_path', required=True, help='path to data')
parser.add_argument('--train_file_list', required=True, help='path to file list of train data')
parser.add_argument('--val_file_list', default='', help='path to file list of val data')
parser.add_argument('--save', default='./logs', help='folder to output model checkpoints')
# model params
parser.add_argument('--retrain', type=str, default='', help='model to load from')
parser.add_argument('--input_dim', type=int, default=0, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
parser.add_argument('--no_logweight_target_sdf', dest='logweight_target_sdf', action='store_false')
# train params
parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels (must be > 1).')
parser.add_argument('--num_iters_per_level', type=int, default=2000, help='#iters before fading in training for next level.')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--max_epoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--save_epoch', type=int, default=1, help='save every nth epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_lr', type=int, default=10, help='decay learning rate by half every n epochs')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
parser.add_argument('--weight_sdf_loss', type=float, default=1.0, help='weight sdf loss vs occ.')
parser.add_argument('--weight_missing_geo', type=float, default=5.0, help='weight missing geometry vs rest of sdf.')
parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
parser.add_argument('--use_loss_masking', dest='use_loss_masking', action='store_true')
parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
parser.add_argument('--scheduler_step_size', type=int, default=0, help='#iters before scheduler step (0 for each epoch)')

parser.set_defaults(no_pass_occ=False, no_pass_feats=False, logweight_target_sdf=True, use_loss_masking=True)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.weight_missing_geo >= 1)
assert( args.num_hierarchy_levels > 1 )
if args.input_dim == 0: # set default values
    args.input_dim = 2 ** (3+args.num_hierarchy_levels)
    #TODO FIX THIS PART
    if '64-64-128' in args.data_path:
        args.input_dim = (128, 64, 64)
    elif '96-96-160' in args.data_path:
        args.input_dim = (160, 96, 96)
    if '64-64-64' in args.data_path:
        args.input_dim = (64, 64, 64)
args.input_nf = 1
UP_AXIS = 0
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

# create model
model = model.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.retrain:
    print('loading model:', args.retrain)
    checkpoint = torch.load(args.retrain)
    args.start_epoch = args.start_epoch if args.start_epoch != 0 else checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict']) #, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
last_epoch = -1 if not args.retrain else args.start_epoch - 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr, gamma=0.5, last_epoch=last_epoch)

# data files
train_files, val_files = data_util.get_train_files(args.data_path, args.train_file_list, args.val_file_list)
_OVERFIT = False
if len(train_files) == 1:
    _OVERFIT = True
    args.use_loss_masking = False
num_overfit_train = 0 if not _OVERFIT else 640
num_overfit_val = 0 if not _OVERFIT else 160
print('#train files = ', len(train_files))
print('#val files = ', len(val_files))
train_dataset = scene_dataloader.SceneDataset(train_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0, num_overfit_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=scene_dataloader.collate)
if len(val_files) > 0:
    val_dataset = scene_dataloader.SceneDataset(val_files, args.input_dim, args.truncation, args.num_hierarchy_levels, 0, num_overfit_val)
    print('val_dataset', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate)

_SPLITTER = ','

def print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, mean_val_losses, mean_val_l1pred, mean_val_l1tgt, mean_val_ious, time, log):
    splitters = ['Epoch: ', ' iter: '] if log is None else ['', ',']
    values = [epoch, iter]
    values.extend(mean_train_losses)
    for h in range(len(mean_train_losses)):
        id = 'total' if h == 0 else str(h-1)
        id = 'sdf' if h + 1 == len(mean_train_losses) else id
        if log is None:
            splitters.append(' loss_train(' + id + '): ')
        else:
            splitters.append(',')
    values.extend([mean_train_l1pred, mean_train_l1tgt])
    if log is None:
        splitters.extend([' train_l1pred: ', ' train_l1tgt: '])
    else:
        splitters.extend([',', ','])
    values.extend(mean_train_ious)
    for h in range(len(mean_train_ious)):
        id = str(h)
        if log is None:
            splitters.append(' iou_train(' + id + '): ')
        else:
            splitters.append(',')
    if mean_val_losses is not None:
        values.extend(mean_val_losses)
        for h in range(len(mean_val_losses)):
            id = 'total' if h == 0 else str(h-1)
            id = 'sdf' if h + 1 == len(mean_val_losses) else id
            if log is None:
                splitters.append(' loss_val(' + id + '): ')
            else:
                splitters.append(',')
        values.extend([mean_val_l1pred, mean_val_l1tgt])
        if log is None:
            splitters.extend([' val_l1pred: ', ' val_l1tgt: '])
        else:
            splitters.extend([',', ','])
        values.extend(mean_val_ious)
        for h in range(len(mean_val_ious)):
            id = str(h)
            if log is None:
                splitters.append(' iou_val(' + id + '): ')
            else:
                splitters.append(',')
    else:
        splitters.extend([''] * (len(mean_train_losses) + len(mean_train_ious) + 2))
        values.extend([''] * (len(mean_train_losses) + len(mean_train_ious) + 2))
    values.append(time)
    if log is None:
        splitters.append(' time: ')
    else:
        splitters.append(',')
    info = ''
    for k in range(len(splitters)):
        if log is None and isinstance(values[k], float):
           info += splitters[k] + '{:.6f}'.format(values[k])
        else:
           info += splitters[k] + str(values[k])
    if log is None:
        print(info, file=sys.stdout)
    else:
        print(info, file=log)

def print_log(log, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, val_losses, val_l1preds, val_l1tgts, val_ious, time):
    train_losses = np.array(train_losses)
    train_l1preds = np.array(train_l1preds)
    train_l1tgts = np.array(train_l1tgts)
    train_ious = np.array(train_ious)
    mean_train_losses = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_losses]
    mean_train_l1pred = -1 if (len(train_l1preds) == 0 or np.all(train_l1preds < 0)) else np.mean(train_l1preds[train_l1preds >= 0])
    mean_train_l1tgt = -1 if (len(train_l1tgts) == 0 or np.all(train_l1tgts < 0))  else np.mean(train_l1tgts[train_l1tgts >= 0])
    mean_train_ious = [(-1 if np.all(x < 0) else np.mean(x[x >= 0])) for x in train_ious]
    mean_val_losses = None
    mean_val_l1pred = None
    mean_val_l1tgt = None
    mean_val_ious = None
    if val_losses:
        val_losses = np.array(val_losses)
        val_l1preds = np.array(val_l1preds)
        val_l1tgts = np.array(val_l1tgts)
        val_ious = np.array(val_ious)
        mean_val_losses = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_losses]
        mean_val_l1pred = -1 if (len(val_l1preds) == 0 or np.all(val_l1preds < 0))  else np.mean(val_l1preds[val_l1preds >= 0])
        mean_val_l1tgt = -1 if (len(val_l1tgts) == 0 or np.all(val_l1tgts < 0))  else np.mean(val_l1tgts[val_l1tgts >= 0])
        mean_val_ious = [-1 if np.all(x < 0) else np.mean(x[x >= 0]) for x in val_ious]
        print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, mean_val_losses, mean_val_l1pred, mean_val_l1tgt, mean_val_ious, time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, mean_val_losses, mean_val_l1pred, mean_val_l1tgt, mean_val_ious, time, log)
    else:
        print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, None, None, None, None, time, None)
        print_log_info(epoch, iter, mean_train_losses, mean_train_l1pred, mean_train_l1tgt, mean_train_ious, None, None, None, None, time, log)
    log.flush()


def get_loss_weights(iter, num_hierarchy_levels, num_iters_per_level, factor_l1_loss):
    weights = np.zeros(num_hierarchy_levels+1, dtype=np.float32)
    cur_level = iter // num_iters_per_level
    if cur_level > num_hierarchy_levels:
        weights.fill(1)
        weights[-1] = factor_l1_loss
        if iter == (num_hierarchy_levels + 1) * num_iters_per_level:
            print('[iter %d] updating loss weights:' % iter, weights)
        return weights
    for level in range(0, cur_level+1):
        weights[level] = 1.0
    step_factor = 20
    fade_amount = max(1.0, min(100, num_iters_per_level//step_factor))
    fade_level = iter % num_iters_per_level
    cur_weight = 0.0
    l1_weight = 0.0
    if fade_level >= num_iters_per_level - fade_amount + step_factor:
        fade_level_step = (fade_level - num_iters_per_level + fade_amount) // step_factor
        cur_weight = float(fade_level_step) / float(fade_amount//step_factor)
    if cur_level+1 < num_hierarchy_levels:
        weights[cur_level+1] = cur_weight
    elif cur_level < num_hierarchy_levels:
        l1_weight = factor_l1_loss * cur_weight
    else:
        l1_weight = 1.0
    weights[-1] = l1_weight
    if iter % num_iters_per_level == 0 or (fade_level >= num_iters_per_level - fade_amount + step_factor and (fade_level - num_iters_per_level + fade_amount) % step_factor == 0):
        print('[iter %d] updating loss weights:' % iter, weights)
    return weights

def train(epoch, iter, dataloader, log_file, output_save):
    train_losses = [ [] for i in range(args.num_hierarchy_levels+2) ]
    train_l1preds = []
    train_l1tgts = []
    train_ious = [ [] for i in range(args.num_hierarchy_levels) ]
    model.train()
    start = time.time()
    
    if args.scheduler_step_size == 0:
        scheduler.step()

    num_batches = len(dataloader)
    for t, sample in enumerate(dataloader):
        loss_weights = get_loss_weights(iter, args.num_hierarchy_levels, args.num_iters_per_level, args.weight_sdf_loss)
        if epoch == args.start_epoch and t == 0:
            print('[iter %d/epoch %d] loss_weights' % (iter, epoch), loss_weights)

        sdfs = sample['sdf']
        if sdfs.shape[0] < args.batch_size:
            continue  # maintain same batch size for training
        inputs = sample['input']
        known = sample['known']
        hierarchy = sample['hierarchy']
        for h in range(len(hierarchy)):
            hierarchy[h] = hierarchy[h].cuda()
        if args.use_loss_masking:
            known = known.cuda()
        inputs[0] = inputs[0].cuda()
        inputs[1] = inputs[1].cuda()
        target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs.cuda(), hierarchy, args.num_hierarchy_levels, args.truncation, args.use_loss_masking, known)

        optimizer.zero_grad()
        output_sdf, output_occs = model(inputs, loss_weights)        
        loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)
        loss.backward()
        optimizer.step()

        output_visual = output_save and t + 2 == num_batches
        compute_pred_occs = (iter % 20 == 0) or output_visual
        if compute_pred_occs:
            pred_occs = [None] * args.num_hierarchy_levels
            for h in range(args.num_hierarchy_levels):
                factor = 2**(args.num_hierarchy_levels-h-1)
                pred_occs[h] = [None] * args.batch_size
                if len(output_occs[h][0]) == 0:
                    continue
                output_occs[h][1] = torch.nn.Sigmoid()(output_occs[h][1][:,0].detach()) > 0.5
                for b in range(args.batch_size):
                    batchmask = output_occs[h][0][:,-1] == b
                    locs = output_occs[h][0][batchmask][:,:-1]
                    vals = output_occs[h][1][batchmask]
                    pred_occs[h][b] = locs[vals.view(-1)]
        train_losses[0].append(loss.item())
        for h in range(args.num_hierarchy_levels):
            train_losses[h+1].append(losses[h])
            target = target_for_occs[h].byte()
            if compute_pred_occs:
                iou = loss_util.compute_iou_sparse_dense(pred_occs[h], target, args.use_loss_masking)
                train_ious[h].append(iou)
        train_losses[args.num_hierarchy_levels+1].append(losses[-1])
        if len(output_sdf[0]) > 0:
            output_sdf = [output_sdf[0].detach(), output_sdf[1].detach()]
        if loss_weights[-1] > 0 and iter % 20 == 0:
            train_l1preds.append(loss_util.compute_l1_predsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, None, False, args.use_loss_masking, known).item())
            train_l1tgts.append(loss_util.compute_l1_tgtsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, args.truncation, args.use_loss_masking, known))

        iter += 1
        if args.scheduler_step_size > 0 and iter % args.scheduler_step_size == 0:
            scheduler.step()
        if iter % 20 == 0:
            took = time.time() - start
            print_log(log_file, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, None, None, None, None, took)
        if iter % 2000 == 0:
            torch.save({'epoch': epoch,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-iter%s-epoch%s.pth' % (iter, epoch)))
        if output_visual:
            vis_pred_sdf = [None] * args.batch_size
            if len(output_sdf[0]) > 0:
                for b in range(args.batch_size):
                    mask = output_sdf[0][:,-1] == b
                    if len(mask) > 0:
                        vis_pred_sdf[b] = [output_sdf[0][mask].cpu().numpy(), output_sdf[1][mask].squeeze().cpu().numpy()]
            inputs = [inputs[0].cpu().numpy(), inputs[1].cpu().numpy()]
            for h in range(args.num_hierarchy_levels):
                for b in range(args.batch_size):
                    if pred_occs[h][b] is not None:
                        pred_occs[h][b] = pred_occs[h][b].cpu().numpy()
            data_util.save_predictions(os.path.join(args.save, 'iter%d-epoch%d' % (iter, epoch), 'train'), sample['name'], inputs, target_for_sdf.cpu().numpy(), [x.cpu().numpy() for x in target_for_occs], vis_pred_sdf, pred_occs, sample['world2grid'].numpy(), args.vis_dfs, args.truncation)

    return train_losses, train_l1preds, train_l1tgts, train_ious, iter, loss_weights


def test(epoch, iter, loss_weights, dataloader, log_file, output_save):
    val_losses = [ [] for i in range(args.num_hierarchy_levels+2) ]
    val_l1preds = []
    val_l1tgts = []
    val_ious = [ [] for i in range(args.num_hierarchy_levels) ]
    model.eval()
    #start = time.time()

    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            sdfs = sample['sdf']
            if sdfs.shape[0] < args.batch_size:
                continue  # maintain same batch size
            inputs = sample['input']
            known = sample['known']
            hierarchy = sample['hierarchy']
            for h in range(len(hierarchy)):
                hierarchy[h] = hierarchy[h].cuda()
            if args.use_loss_masking:
                known = known.cuda()
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs.cuda(), hierarchy, args.num_hierarchy_levels, args.truncation, args.use_loss_masking, known)

            output_sdf, output_occs = model(inputs, loss_weights)
            loss, losses = loss_util.compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, args.truncation, args.logweight_target_sdf, args.weight_missing_geo, inputs[0], args.use_loss_masking, known)

            output_visual = output_save and t + 2 == num_batches
            compute_pred_occs = (t % 20 == 0) or output_visual
            if compute_pred_occs:
                pred_occs = [None] * args.num_hierarchy_levels
                for h in range(args.num_hierarchy_levels):
                    factor = 2**(args.num_hierarchy_levels-h-1)
                    pred_occs[h] = [None] * args.batch_size
                    if len(output_occs[h][0]) == 0:
                        continue
                    for b in range(args.batch_size):
                        batchmask = output_occs[h][0][:,-1] == b
                        locs = output_occs[h][0][batchmask][:,:-1]
                        vals = torch.nn.Sigmoid()(output_occs[h][1][:,0].detach()[batchmask]) > 0.5
                        pred_occs[h][b] = locs[vals.view(-1)]
            val_losses[0].append(loss.item())
            for h in range(args.num_hierarchy_levels):
                val_losses[h+1].append(losses[h])
                target = target_for_occs[h].byte()
                if compute_pred_occs:
                    iou = loss_util.compute_iou_sparse_dense(pred_occs[h], target, args.use_loss_masking)
                    val_ious[h].append(iou)
            val_losses[args.num_hierarchy_levels+1].append(losses[-1])
            if len(output_sdf[0]) > 0:
                output_sdf = [output_sdf[0].detach(), output_sdf[1].detach()]
            if loss_weights[-1] > 0 and t % 20 == 0:
                val_l1preds.append(loss_util.compute_l1_predsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, None, False, args.use_loss_masking, known).item())
                val_l1tgts.append(loss_util.compute_l1_tgtsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, args.truncation, args.use_loss_masking, known))
            if output_visual:
                vis_pred_sdf = [None] * args.batch_size
                if len(output_sdf[0]) > 0:
                    for b in range(args.batch_size):
                        mask = output_sdf[0][:,-1] == b
                        if len(mask) > 0:
                            vis_pred_sdf[b] = [output_sdf[0][mask].cpu().numpy(), output_sdf[1][mask].squeeze().cpu().numpy()]
                inputs = [inputs[0].cpu().numpy(), inputs[1].cpu().numpy()]
                for h in range(args.num_hierarchy_levels):
                    for b in range(args.batch_size):
                        if pred_occs[h][b] is not None:
                            pred_occs[h][b] = pred_occs[h][b].cpu().numpy()
                data_util.save_predictions(os.path.join(args.save, 'iter%d-epoch%d' % (iter, epoch), 'val'), sample['name'], inputs, target_for_sdf.cpu().numpy(), [x.cpu().numpy() for x in target_for_occs], vis_pred_sdf, pred_occs, sample['world2grid'], args.vis_dfs, args.truncation)

    #took = time.time() - start
    return val_losses, val_l1preds, val_l1tgts, val_ious


def main():
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    elif not _OVERFIT:
        raw_input('warning: save dir %s exists, press key to delete and continue' % args.save)

    data_util.dump_args_txt(args, os.path.join(args.save, 'args.txt'))
    log_file = open(os.path.join(args.save, 'log.csv'), 'w')
    headers = ['epoch','iter','train_loss(total)']
    for h in range(args.num_hierarchy_levels):
        headers.append('train_loss(' + str(h) + ')')
    headers.extend(['train_loss(sdf)', 'train_l1-pred', 'train_l1-tgt'])
    for h in range(args.num_hierarchy_levels):
        headers.append('train_iou(' + str(h) + ')')
    headers.extend(['time'])
    log_file.write(_SPLITTER.join(headers) + '\n')
    log_file.flush()

    has_val = len(val_files) > 0
    log_file_val = None
    if has_val:
        headers = headers[:-1]
        headers.append('val_loss(total)')
        for h in range(args.num_hierarchy_levels):
            headers.append('val_loss(' + str(h) + ')')
        headers.extend(['val_loss(sdf)', 'val_l1-pred', 'val_l1-tgt'])
        for h in range(args.num_hierarchy_levels):
            headers.append('val_iou(' + str(h) + ')')
        headers.extend(['time'])
        log_file_val = open(os.path.join(args.save, 'log_val.csv'), 'w')
        log_file_val.write(_SPLITTER.join(headers) + '\n')
        log_file_val.flush()
    # start training
    print('starting training...')
    iter = args.start_epoch * (len(train_dataset) // args.batch_size)
    for epoch in range(args.start_epoch, args.max_epoch):
        start = time.time()

        train_losses, train_l1preds, train_l1tgts, train_ious, iter, loss_weights = train(epoch, iter, train_dataloader, log_file, output_save=(epoch % args.save_epoch == 0))
        if has_val:
            val_losses, val_l1preds, val_l1tgts, val_ious = test(epoch, iter, loss_weights, val_dataloader, log_file_val, output_save=(epoch % args.save_epoch == 0))

        took = time.time() - start
        if has_val:
            print_log(log_file_val, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, val_losses, val_l1preds, val_l1tgts, val_ious, took)
        else:
            print_log(log_file, epoch, iter, train_losses, train_l1preds, train_l1tgts, train_ious, None, None, None, None, took)
        torch.save({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(args.save, 'model-epoch-%s.pth' % epoch))
    log_file.close()
    if has_val:
        log_file_val.close()



if __name__ == '__main__':
    main()


