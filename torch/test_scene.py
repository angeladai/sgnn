from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import random
import torch
import numpy as np
import gc

import data_util
import scene_dataloader
import model_sparse as model
import loss as loss_util

# python test_scene.py --gpu 0 --input_data_path /mnt/raid/adai/data/matterport/mp_sdf_vox_5cm_incomplete --target_data_path /mnt/raid/adai/data/matterport/mp_sdf_vox_5cm --num_hierarchy_levels 3 --test_file_list filelists/mp_rooms_1.txt --output output/mp/debug


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--input_data_path', required=True, help='path to input data')
parser.add_argument('--target_data_path', required=True, help='path to target data')
parser.add_argument('--test_file_list', required=True, help='path to file list of test data')
parser.add_argument('--model_path', required=True, help='path to model to test')
parser.add_argument('--output', default='./output', help='folder to output predictions')
# model params
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
parser.add_argument('--logweight_target_sdf', dest='logweight_target_sdf', action='store_true')
parser.add_argument('--no_logweight_target_sdf', dest='logweight_target_sdf', action='store_false')
# test params
parser.add_argument('--num_hierarchy_levels', type=int, default=3, help='#hierarchy levels.')
parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--max_to_process', type=int, default=150, help='max num to process')
parser.add_argument('--weight_sdf_loss', type=float, default=1.0, help='weight sdf loss vs occ.')
parser.add_argument('--weight_missing_geo', type=float, default=1.0, help='weight missing geometry vs rest of sdf.')
parser.add_argument('--vis_dfs', type=int, default=0, help='use df (iso 1) to visualize')
parser.add_argument('--use_loss_masking', dest='use_loss_masking', action='store_true')
parser.add_argument('--no_loss_masking', dest='use_loss_masking', action='store_false')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(no_pass_occ=False, no_pass_feats=False, use_loss_masking=True, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.weight_missing_geo >= 1)
args.input_nf = 1
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
UP_AXIS = 0 # z is 0th 


# create model
model = model.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense)
if not args.cpu:
    model = model.cuda()
checkpoint = torch.load(args.model_path)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(torch.load(args.model_path))
print('loaded model:', args.model_path)


_SPLITTER = ','


def test(loss_weights, dataloader, output_pred, output_vis, num_to_vis):
    model.eval()
    missing = []

    num_proc = 0
    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs = sample['input']
            sdfs = sample['sdf']
            known = sample['known']
            hierarchy = sample['hierarchy']
            input_dim = np.array(sdfs.shape[2:])
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_proc, args.max_to_process, sample['name'], input_dim[0], input_dim[1], input_dim[2]))
            sys.stdout.flush()
            target_for_sdf, target_for_occs, target_for_hier = loss_util.compute_targets(sdfs, hierarchy, args.num_hierarchy_levels, args.truncation, args.use_loss_masking, known)
            hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
            model.update_sizes(input_dim, input_dim // hierarchy_factor)
            try:
                if not args.cpu:
                    inputs[1] = inputs[1].cuda()
                    target_for_sdf = target_for_sdf.cuda()
                    for h in range(len(target_for_occs)):
                        target_for_occs[h] = target_for_occs[h].cuda()
                        target_for_hier[h] = target_for_hier[h].cuda()
                output_sdf, output_occs = model(inputs, loss_weights)
            except:
                print('exception at %s' % sample['name'])
                gc.collect()
                missing.extend(sample['name'])
                continue
            # remove padding
            dims = sample['orig_dims'][0]
            mask = (output_sdf[0][:,0] < dims[0]) & (output_sdf[0][:,1] < dims[1]) & (output_sdf[0][:,2] < dims[2])
            output_sdf[0] = output_sdf[0][mask]
            output_sdf[1] = output_sdf[1][mask]
            for h in range(len(output_occs)):
                dims = target_for_occs[h].shape[2:]
                mask = (output_occs[h][0][:,0] < dims[0]) & (output_occs[h][0][:,1] < dims[1]) & (output_occs[h][0][:,2] < dims[2])
                output_occs[h][0] = output_occs[h][0][mask]
                output_occs[h][1] = output_occs[h][1][mask]

            # save prediction files
            data_util.save_predictions_to_file(output_sdf[0][:,:3].cpu().numpy(), output_sdf[1].cpu().numpy(), os.path.join(output_pred, sample['name'][0] + '.pred'))

            try:
                pred_occs = [None] * args.num_hierarchy_levels
                for h in range(args.num_hierarchy_levels):
                    pred_occs[h] = [None]
                    if len(output_occs[h][0]) == 0:
                        continue
                    locs = output_occs[h][0][:,:-1]
                    vals = torch.nn.Sigmoid()(output_occs[h][1][:,0].detach()) > 0.5
                    pred_occs[h][0] = locs[vals.view(-1)]
            except:
                print('exception at %s' % sample['name'])
                gc.collect()
                missing.extend(sample['name'])
                continue

            num_proc += 1
            if num_vis < num_to_vis:
                num = min(num_to_vis - num_vis, 1)
                vis_pred_sdf = [None] * num
                if len(output_sdf[0]) > 0:
                    for b in range(num):
                        mask = output_sdf[0][:,-1] == b
                        if len(mask) > 0:
                            vis_pred_sdf[b] = [output_sdf[0][mask].cpu().numpy(), output_sdf[1][mask].squeeze().cpu().numpy()]
                inputs = [inputs[0].numpy(), inputs[1].cpu().numpy()]
                data_util.save_predictions(output_vis, np.arange(num), inputs, target_for_sdf.cpu().numpy(), [x.cpu().numpy() for x in target_for_occs], vis_pred_sdf, pred_occs, sample['world2grid'], args.vis_dfs, args.truncation)
                num_vis += 1
            gc.collect()

    sys.stdout.write('\n')
    print('missing', missing)


def main():
    # data files
    test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
    if len(test_files) > args.max_to_process:
        test_files = test_files[:args.max_to_process]
    else:
        args.max_to_process = len(test_files)
    random.seed(42)
    random.shuffle(test_files)
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, args.max_input_height, 0, args.target_data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate)

    if os.path.exists(args.output):
        raw_input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    output_vis_path = os.path.join(args.output, 'vis') if not args.vis_dfs else os.path.join(args.output, 'vis-dfs')
    if not os.path.exists(output_vis_path):
        os.makedirs(output_vis_path)
    output_pred_path = os.path.join(args.output, 'pred')
    if not os.path.exists(output_pred_path):
        os.makedirs(output_pred_path)

    # start testing
    print('starting testing...')
    loss_weights = np.ones(args.num_hierarchy_levels+1, dtype=np.float32)
    loss_weights[-1] = args.weight_sdf_loss
    test(loss_weights, test_dataloader, output_pred_path, output_vis_path, args.num_to_vis)


if __name__ == '__main__':
    main()


