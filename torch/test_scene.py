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
import model
import loss as loss_util

# python test_scene.py --gpu 0 --input_data_path ./data/mp_sdf_vox_2cm_input --target_data_path ./data/mp_sdf_vox_2cm_target --test_file_list filelists/mp_rooms_1.txt --output output/mp


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
parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# test params
parser.add_argument('--max_to_vis', type=int, default=10, help='max num to vis')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
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
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)


def test(loss_weights, dataloader, output_vis, num_to_vis):
    model.eval()

    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            inputs = sample['input']
            input_dim = np.array(sample['sdf'].shape[2:])
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_vis, num_to_vis, sample['name'], input_dim[0], input_dim[1], input_dim[2]))
            sys.stdout.flush()
            hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
            model.update_sizes(input_dim, input_dim // hierarchy_factor)
            try:
                if not args.cpu:
                    inputs[1] = inputs[1].cuda()
                output_sdf, output_occs = model(inputs, loss_weights)
            except:
                print('exception at %s' % sample['name'])
                gc.collect()
                continue

            # remove padding
            dims = sample['orig_dims'][0]
            mask = (output_sdf[0][:,0] < dims[0]) & (output_sdf[0][:,1] < dims[1]) & (output_sdf[0][:,2] < dims[2])
            output_sdf[0] = output_sdf[0][mask]
            output_sdf[1] = output_sdf[1][mask]
            mask = (inputs[0][:,0] < dims[0]) & (inputs[0][:,1] < dims[1]) & (inputs[0][:,2] < dims[2])
            inputs[0] = inputs[0][mask]
            inputs[1] = inputs[1][mask]
            vis_pred_sdf = [None]
            if len(output_sdf[0]) > 0:
                vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
            inputs = [inputs[0].numpy(), inputs[1].cpu().numpy()]
            data_util.save_predictions(output_vis, sample['name'], inputs, None, None, vis_pred_sdf, None, sample['world2grid'], args.truncation)
            num_vis += 1
            if num_vis >= num_to_vis:
                break
    sys.stdout.write('\n')


def main():
    # data files
    test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
    if len(test_files) > args.max_to_vis:
        test_files = test_files[:args.max_to_vis]
    else:
        args.max_to_vis = len(test_files)
    random.seed(42)
    random.shuffle(test_files)
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, args.max_input_height, 0, args.target_data_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate)

    if os.path.exists(args.output):
        raw_input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start testing
    print('starting testing...')
    loss_weights = np.ones(args.num_hierarchy_levels+1, dtype=np.float32)
    test(loss_weights, test_dataloader, args.output, args.max_to_vis)


if __name__ == '__main__':
    main()


