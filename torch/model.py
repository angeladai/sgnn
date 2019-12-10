
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sparseconvnet as scn

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

FSIZE0 = 3
FSIZE1 = 2

class SparseEncoderLayer(nn.Module):
    def __init__(self, nf_in, nf, input_sparsetensor, return_sparsetensor, max_data_size):
        nn.Module.__init__(self)
        data_dim = 3
        self.nf_in = nf_in
        self.nf = nf
        self.input_sparsetensor = input_sparsetensor
        self.return_sparsetensor = return_sparsetensor
        self.max_data_size = max_data_size
        if not self.input_sparsetensor:
            self.p0 = scn.InputLayer(data_dim, self.max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.Sequential()
        self.p2.add(scn.ConcatTable()
                    .add(scn.Identity())
                    .add(scn.Sequential()
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False))
                        .add(scn.BatchNormReLU(nf))
                        .add(scn.SubmanifoldConvolution(data_dim, nf, nf, FSIZE0, False)))
                ).add(scn.AddTable())
        self.p2.add(scn.BatchNormReLU(nf))
        # downsample space by factor of 2
        self.p3 = scn.Sequential().add(scn.Convolution(data_dim, nf, nf, FSIZE1, 2, False))
        self.p3.add(scn.BatchNormReLU(nf))
        if not self.return_sparsetensor:
            self.p4 = scn.SparseToDense(data_dim, nf)

    def forward(self,x):
        if not self.input_sparsetensor:
            x = self.p0(x)
        #print('x', x.spatial_size, x.metadata.getSpatialLocations(x.spatial_size).shape, torch.max(x.metadata.getSpatialLocations(x.spatial_size)[:,:-1]).item(), x.features.shape)
        x = self.p1(x)
        #print('x(p1)', x.spatial_size, x.metadata.getSpatialLocations(x.spatial_size).shape, torch.max(x.metadata.getSpatialLocations(x.spatial_size)[:,:-1]).item(), x.features.shape)
        x = self.p2(x)
        ft2 = x
        #print('x(p2)', x.spatial_size, x.metadata.getSpatialLocations(x.spatial_size).shape, torch.max(x.metadata.getSpatialLocations(x.spatial_size)[:,:-1]).item(), x.features.shape)
        x = self.p3(x)
        #print('x(p3)', x.spatial_size, x.metadata.getSpatialLocations(x.spatial_size).shape, torch.max(x.metadata.getSpatialLocations(x.spatial_size)[:,:-1]).item(), x.features.shape)
        if self.return_sparsetensor:
            #print('sparse encode output:', x.metadata.getSpatialLocations(x.spatial_size).shape, x.features.shape)
            return x, [ft2]
        else: # densify
            ft3 = x
            x = self.p4(x)
            #print('sparse encode output:', x.shape)
            return x, [ft2, ft3]

class TSDFEncoder(nn.Module):
    def __init__(self, nf_in, nf_per_level, nf_out, use_skip_sparse, use_skip_dense, input_volume_size):
        nn.Module.__init__(self)
        assert (type(nf_per_level) is list)
        data_dim = 3
        self.use_skip_sparse = use_skip_sparse
        self.use_skip_dense = use_skip_dense
        #self.use_bias = True
        self.use_bias = False
        modules = []
        volume_sizes = [(np.array(input_volume_size) // (k + 1)).tolist() for k in range(len(nf_per_level))]
        for level in range(len(nf_per_level)):
            nf_in = nf_in if level == 0 else nf_per_level[level-1]
            input_sparsetensor = level > 0
            return_sparsetensor = (level < len(nf_per_level) - 1)
            modules.append(SparseEncoderLayer(nf_in, nf_per_level[level], input_sparsetensor, return_sparsetensor, volume_sizes[level]))
        self.process_sparse = nn.Sequential(*modules)
        nf = nf_per_level[-1]
        # 16 -> 8
        nf0 = nf*3 // 2
        self.encode_dense0 = nn.Sequential(
            nn.Conv3d(nf, nf0, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf0),
            nn.ReLU(True)
        )
        # 8 -> 4
        nf1 = nf*2
        self.encode_dense1 = nn.Sequential(
            nn.Conv3d(nf0, nf1, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf1),
            nn.ReLU(True)
        )
        # 4 -> 4
        nf2 = nf1
        self.bottleneck_dense2 = nn.Sequential(
            nn.Conv3d(nf1, nf2, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf2),
            nn.ReLU(True)
        )
        # 4 -> 8
        nf3 = nf2 if not self.use_skip_dense else nf1+nf2
        nf4 = nf3 // 2
        self.decode_dense3 = nn.Sequential(
            nn.ConvTranspose3d(nf3, nf4, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf4),
            nn.ReLU(True)
        )
        # 8 -> 16
        if self.use_skip_dense:
            nf4 += nf0
        nf5 = nf4 // 2
        self.decode_dense4 = nn.Sequential(
            nn.ConvTranspose3d(nf4, nf5, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf5),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Conv3d(nf5, nf_out, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU(True)
        )
        # occ prediction
        self.occpred = nn.Sequential(
            nn.Conv3d(nf_out, 1, kernel_size=1, bias=self.use_bias)
        )
        self.sdfpred = nn.Sequential(
            nn.Conv3d(nf_out, 1, kernel_size=1, bias=self.use_bias)
        )
        # debug stats
        params_encodesparse = count_num_model_params(self.process_sparse)
        params_encodedense = count_num_model_params(self.encode_dense0) + count_num_model_params(self.encode_dense0) + count_num_model_params(self.encode_dense1) + count_num_model_params(self.bottleneck_dense2)
        params_decodedense = count_num_model_params(self.decode_dense3) + count_num_model_params(self.decode_dense4) + count_num_model_params(self.final) + count_num_model_params(self.occpred)
        print('[TSDFEncoder] params encode sparse', params_encodesparse)
        print('[TSDFEncoder] params encode dense', params_encodedense)
        print('[TSDFEncoder] params decode dense', params_decodedense)

    def forward(self,x):
        feats_sparse = []
        for k in range(len(self.process_sparse)):
            x, ft = self.process_sparse[k](x)
            if self.use_skip_sparse:
                feats_sparse.extend(ft)
        
        enc0 = self.encode_dense0(x)
        enc1 = self.encode_dense1(enc0)
        bottleneck = self.bottleneck_dense2(enc1)
        if self.use_skip_dense:
            dec0 = self.decode_dense3(torch.cat([bottleneck, enc1], 1))
        else:
            dec0 = self.decode_dense3(bottleneck)
        if self.use_skip_dense:
            x = self.decode_dense4(torch.cat([dec0, enc0], 1))
        else:
            x = self.decode_dense4(dec0)
        x = self.final(x)
        occ = self.occpred(x)
        sdf = self.sdfpred(x)
        out = torch.cat([occ, sdf],1)
        return x, out, feats_sparse

class Refinement(nn.Module):
    def __init__(self, nf_in, nf, pass_occ, pass_feats, max_data_size, truncation=3):
        nn.Module.__init__(self)
        data_dim = 3
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        self.nf_in = nf_in
        self.nf = nf
        self.truncation = truncation
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf*3)
        self.p4 = scn.OutputLayer(data_dim)

        # upsampled 
        self.n0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.n1 = scn.SubmanifoldConvolution(data_dim, nf*3, nf, filter_size=FSIZE0, bias=False)
        self.n2 = scn.BatchNormReLU(nf)
        self.n3 = scn.OutputLayer(data_dim)
        self.linear = nn.Linear(nf, 1)
        self.linearsdf = nn.Linear(nf, 1)

    def to_next_level_locs(self, locs, feats): # upsample factor of 2 predictions
        assert(len(locs.shape) == 2)
        data_dim = locs.shape[-1] - 1 # assumes batch mode 
        offsets = torch.nonzero(torch.ones(2,2,2)).long() # 8 x 3 
        locs_next = locs.unsqueeze(1).repeat(1, 8, 1)
        locs_next[:,:,:data_dim] *= 2
        locs_next[:,:,:data_dim] += offsets
        #print('locs', locs.shape, locs.type())
        #print('locs_next', locs_next.shape, locs_next.type())
        #print('locs_next.view(-1,4)[:20]', locs_next.view(-1,4)[:20])
        feats_next = feats.unsqueeze(1).repeat(1, 8, 1) # TODO: CUSTOM TRILERP HERE???
        #print('feats', feats.shape, feats.type())
        #print('feats_next', feats_next.shape, feats_next.type())
        #print('feats_next.view(-1,feats.shape[-1])[:20,:5]', feats_next.view(-1,feats.shape[-1])[:20,:5])
        #raw_input('sdlfkj')
        return locs_next.view(-1, locs.shape[-1]), feats_next.view(-1, feats.shape[-1])

    def forward(self,x):
        input_locs = x[0]
        if len(input_locs) == 0:
            return [[],[]],[[],[]]
        #x=self.sparseModel(x)
        #print('x(sparse)', x.shape)

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        locs_unfilt, feats = self.to_next_level_locs(input_locs, x)

        x = self.n0([locs_unfilt, feats])
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)

        # predict occupancy
        out = self.linear(x)
        sdf = self.linearsdf(x)
        # mask out for next level processing
        mask = (nn.Sigmoid()(out) > 0.5).view(-1)
        #print('x', x.type(), x.shape, torch.min(x).item(), torch.max(x).item())
        #print('locs_unfilt', locs_unfilt.type(), locs_unfilt.shape, torch.min(locs_unfilt).item(), torch.max(locs_unfilt).item())
        #print('out', out.type(), out.shape, torch.min(out).item(), torch.max(out).item())
        #print('mask', mask.type(), mask.shape, torch.sum(mask).item())
        locs = locs_unfilt[mask]
        
        out = torch.cat([out, sdf],1)
        if self.pass_feats and self.pass_occ:
            feats = torch.cat([x[mask], out[mask]], 1)
        elif self.pass_feats:
            feats = x[mask]
        elif self.pass_occ:
            feats = out[mask]
        return [locs, feats], [locs_unfilt, out]

class SurfacePrediction(nn.Module):
    def __init__(self, nf_in, nf, nf_out, max_data_size):
        nn.Module.__init__(self)
        data_dim = 3
        self.p0 = scn.InputLayer(data_dim, max_data_size, mode=0)
        self.p1 = scn.SubmanifoldConvolution(data_dim, nf_in, nf, filter_size=FSIZE0, bias=False)
        self.p2 = scn.FullyConvolutionalNet(data_dim, reps=1, nPlanes=[nf, nf, nf], residual_blocks=True) #nPlanes=[nf, nf*2, nf*2], residual_blocks=True)
        self.p3 = scn.BatchNormReLU(nf*3)
        self.p4 = scn.OutputLayer(data_dim)
        self.linear = nn.Linear(nf*3, nf_out)
    def forward(self,x):
        if len(x[0]) == 0:
            return [], []
        #x=self.sparseModel(x)
        #print('x(sparse)', x.shape)

        x = self.p0(x)
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        
        x=self.linear(x)
        return x


# ==== model ==== #
class GenModel(nn.Module):
    def __init__(self, encoder_dim, input_dim, input_nf, nf_coarse, nf, num_hierarchy_levels, pass_occ, pass_feats, use_skip_sparse, use_skip_dense, truncation=3):
        nn.Module.__init__(self)
        self.truncation = truncation
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        # encoder
        if not isinstance(input_dim, (list, tuple, np.ndarray)):
            input_dim = [input_dim, input_dim, input_dim]
        #self.nf_per_level = [encoder_dim*(k+1) for k in range(num_hierarchy_levels-1)]
        self.nf_per_level = [int(encoder_dim*(1+float(k)/(num_hierarchy_levels-2))) for k in range(num_hierarchy_levels-1)] if num_hierarchy_levels > 2 else [encoder_dim]*(num_hierarchy_levels-1)
        self.use_skip_sparse = use_skip_sparse
        self.encoder = TSDFEncoder(input_nf, self.nf_per_level, nf_coarse, self.use_skip_sparse, use_skip_dense, input_volume_size=input_dim)

        self.refine_sizes = [(np.array(input_dim) // (pow(2,k))).tolist() for k in range(num_hierarchy_levels-1)][::-1]
        self.nf_per_level.append(self.nf_per_level[-1])
        print('#params encoder', count_num_model_params(self.encoder))

        # sparse prediction
        self.data_dim = 3
        self.refinement = scn.Sequential()
        for h in range(1, num_hierarchy_levels):
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[num_hierarchy_levels - h]
            if pass_occ:
                nf_in += 2
            if pass_feats:
                nf_in += (nf_coarse if h == 1 else nf)
            self.refinement.add(Refinement(nf_in, nf, pass_occ, pass_feats, self.refine_sizes[h-1], truncation=self.truncation))
        print('#params refinement', count_num_model_params(self.refinement))
        self.PRED_SURF = True
        if self.PRED_SURF:
            nf_in = 0 if not self.use_skip_sparse else self.nf_per_level[0]
            nf_out = 1
            if pass_occ:
                nf_in += 2
            if pass_feats:
                nf_in += nf
            self.surfacepred = SurfacePrediction(nf_in, nf, nf_out, self.refine_sizes[-1])
            print('#params surfacepred', count_num_model_params(self.surfacepred))
    def dense_coarse_to_sparse(self, coarse_feats, coarse_occ, truncation):
        nf = coarse_feats.shape[1]
        batch_size = coarse_feats.shape[0]
        # sparse locations
        locs_unfilt = torch.nonzero(torch.ones([coarse_occ.shape[2], coarse_occ.shape[3], coarse_occ.shape[4]])).unsqueeze(0).repeat(coarse_occ.shape[0], 1, 1).view(-1, 3)
        batches = torch.arange(coarse_occ.shape[0]).to(locs_unfilt.device).unsqueeze(1).repeat(1, coarse_occ.shape[2]*coarse_occ.shape[3]*coarse_occ.shape[4]).view(-1, 1)
        locs_unfilt = torch.cat([locs_unfilt, batches], 1)
        mask = nn.Sigmoid()(coarse_occ[:,0,:,:,:]) > 0.5
        if self.pass_feats:
            feats_feats = coarse_feats.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, nf)
            feats_feats = feats_feats[mask.view(batch_size, -1)]
        coarse_occ = coarse_occ.permute(0, 2, 3, 4, 1).contiguous()
        if self.pass_occ:
            occ_feats = coarse_occ[mask]
        if self.pass_occ and self.pass_feats:
            feats = torch.cat([occ_feats, feats_feats], 1)
        elif self.pass_occ:
            feats = occ_feats
        elif self.pass_feats:
            feats = feats_feats
        locs = locs_unfilt[mask.view(-1)]
        return locs, feats, [locs_unfilt, coarse_occ.view(-1, 2)]

    def concat_skip(self, x_from, x_to, spatial_size, batch_size):
        locs_from = x_from[0]
        locs_to = x_to[0]
        if len(locs_from) == 0 or len(locs_to) == 0:
            return x_to
        # python implementation here
        locs_from = (locs_from[:,0] * spatial_size[1] * spatial_size[2] + locs_from[:,1] * spatial_size[2] + locs_from[:,2]) * batch_size + locs_from[:,3]
        locs_to = (locs_to[:,0] * spatial_size[1] * spatial_size[2] + locs_to[:,1] * spatial_size[2] + locs_to[:,2]) * batch_size + locs_to[:,3]
        indicator_from = torch.zeros(spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size, dtype=torch.long, device=locs_from.device)
        indicator_to = indicator_from.clone()
        indicator_from[locs_from] = torch.arange(locs_from.shape[0], device=locs_from.device) + 1
        indicator_to[locs_to] = torch.arange(locs_to.shape[0], device=locs_to.device) + 1
        inds = torch.nonzero((indicator_from > 0) & (indicator_to > 0)).squeeze()
        feats_from = x_from[1].new_zeros(x_to[1].shape[0], x_from[1].shape[1])
        if inds.shape[0] > 0:
            feats_from[indicator_to[inds]-1] = x_from[1][indicator_from[inds]-1]
        x_to[1] = torch.cat([x_to[1], feats_from], 1)
        return x_to

    def update_sizes(self, input_max_dim, refine_max_dim):
        print('[model:update_sizes]', input_max_dim, refine_max_dim)
        if not isinstance(input_max_dim, (list, tuple, np.ndarray)):
            input_max_dim = [input_max_dim, input_max_dim, input_max_dim]
        if not isinstance(refine_max_dim, (list, tuple, np.ndarray)):
            refine_max_dim = [refine_max_dim, refine_max_dim, refine_max_dim]
        for k in range(3):
            self.encoder.process_sparse[0].p0.spatial_size[k] = input_max_dim[k]
            for h in range(len(self.refinement)):
                self.refinement[h].p0.spatial_size[k] = refine_max_dim[k]
                refine_max_dim *= 2
                self.refinement[h].n0.spatial_size[k] = refine_max_dim[k]
            self.surfacepred.p0.spatial_size[k] = refine_max_dim[k]

    def forward(self, x, loss_weights):
        outputs = []
        #print('[model] x', x[0].shape, x[1].shape, torch.max(x[0][:,0]).item(), torch.max(x[0][:,1]).item(), torch.max(x[0][:,2]).item())
        # encode
        x, out, feats_sparse = self.encoder(x)
        batch_size = x.shape[0]
        if self.use_skip_sparse:
            for k in range(len(feats_sparse)):
                #print('[model] feats_sparse[%d]' % k, feats_sparse[k].spatial_size)
                feats_sparse[k] = ([feats_sparse[k].metadata.getSpatialLocations(feats_sparse[k].spatial_size), scn.OutputLayer(3)(feats_sparse[k])], feats_sparse[k].spatial_size)
        locs, feats, out = self.dense_coarse_to_sparse(x, out, truncation=3)
        outputs.append(out)
        #print('locs, feats', locs.shape, locs.type(), feats.shape, feats.type(), x.shape)
        #raw_input('sdflkj')

        x_sparse = [locs, feats]
        for h in range(len(self.refinement)):
            if loss_weights[h+1] > 0:
                if self.use_skip_sparse:
                    x_sparse = self.concat_skip(feats_sparse[len(self.refinement)-h][0], x_sparse, feats_sparse[len(self.refinement)-h][1], batch_size)
                #print('[model] refine(%d) x_sparse(input)' % h, x_sparse[0].shape, torch.min(x_sparse[0]).item(), torch.max(x_sparse[0]).item())
                x_sparse, occ = self.refinement[h](x_sparse)
                outputs.append(occ)
                #print('[model] refine(%d) x_sparse' % h, x_sparse[0].shape, torch.min(x_sparse[0]).item(), torch.max(x_sparse[0]).item())
            else:
                outputs.append([[],[]])
        # surface prediction
        locs = x_sparse[0]
        if self.PRED_SURF and loss_weights[-1] > 0:
            if self.use_skip_sparse:
                x_sparse = self.concat_skip(feats_sparse[0][0], x_sparse, feats_sparse[0][1], batch_size)
            x_sparse = self.surfacepred(x_sparse)
            #print('[model] surfpred x_sparse', x_sparse.shape)
            # #DEBUG SANITY - check batching same
            # print('locs', locs.shape)
            # print('x_sparse', x_sparse.shape)
            # for b in [0,1,2]:
            #     batchmask = locs[:,3] == b
            #     batchlocs = locs[batchmask]
            #     batchfeats = x_sparse[batchmask]
            #     print('[%d] batchlocs' % b, batchlocs.shape, torch.min(batchlocs[:,:-1]).item(), torch.max(batchlocs[:,:-1]).item(), torch.sum(batchlocs[:,:-1]).item())
            #     print('[%d] batchfeats' % b, batchfeats.shape, torch.min(batchfeats).item(), torch.max(batchfeats).item(), torch.sum(batchfeats).item())
            # raw_input('sdlfkj')
            # #DEBUG SANITY - check batching same
            return [locs, x_sparse], outputs
        return [[],[]], outputs


if __name__ == '__main__':
    use_cuda = True

    model = GenModel(encoder_dim=8, input_dim=(128,128,128), input_nf=4, nf_coarse=16, nf=16, num_hierarchy_levels=4, pass_occ=True, pass_feats=True, use_skip_sparse=1, use_skip_dense=1)
    # batch size 10 -- all batches are identical
    locs = torch.randint(low=0, high=128, size=(100, 3)).unsqueeze(0).repeat(10, 1, 1).view(-1, 3)
    batches = torch.ones(locs.shape[0]).long()
    for b in range(10):
        batches[b*100:(b+1)*100] = b
    batches = batches.unsqueeze(1)
    locs = torch.cat([locs, batches], 1)
    feats = torch.rand(100).unsqueeze(0).repeat(10, 1).view(-1, 1).float()
    print('locs', locs.shape, torch.min(locs).item(), torch.max(locs).item())
    print('feats', feats.shape, torch.min(feats).item(), torch.max(feats).item())
    if use_cuda:
        model = model.cuda()
        locs = locs.cuda()
        feats = feats.cuda()
    output_sdf, output_occs = model([locs, feats], loss_weights=[1, 1, 1, 1])
    print('output_sdf[0]', output_sdf[0].shape, torch.min(output_sdf[0]).item(), torch.max(output_sdf[0]).item())
    print('output_sdf[1]', output_sdf[1].shape, torch.min(output_sdf[1]).item(), torch.max(output_sdf[1]).item())

