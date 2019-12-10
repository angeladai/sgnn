
import numpy as np
import torch
import torch.nn.functional as F

import sparseconvnet as scn

import data_util

UNK_THRESH = 2
#UNK_THRESH = 3

UNK_ID = -1

def compute_targets(target, hierarchy, num_hierarchy_levels, truncation, use_loss_masking, known):
    assert(len(target.shape) == 5)
    target_for_occs = [None] * num_hierarchy_levels
    target_for_hier = [None] * num_hierarchy_levels
    target_for_sdf = data_util.preprocess_sdf_pt(target, truncation)
    known_mask = None
    target_for_hier[-1] = target.clone()
    target_occ = (torch.abs(target_for_sdf) < truncation).float()
    if use_loss_masking:
        target_occ[known >= UNK_THRESH] = UNK_ID
    target_for_occs[-1] = target_occ

    factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        target_for_occs[h] = torch.nn.MaxPool3d(kernel_size=2)(target_for_occs[h+1])
        target_for_hier[h] = data_util.preprocess_sdf_pt(hierarchy[h], truncation)
        factor *= 2
    return target_for_sdf, target_for_occs, target_for_hier

# note: weight_missing_geo must be > 1
def compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation):
    num_hierarchy_levels = len(target_for_occs)
    weights = [None] * num_hierarchy_levels
    dims = target_for_occs[-1].shape[2:]
    flatlocs = input_locs[:,3]*dims[0]*dims[1]*dims[2] + input_locs[:,0]*dims[1]*dims[2] + input_locs[:,1]*dims[2] + input_locs[:,2]
    weights[-1] = torch.ones(target_for_occs[-1].shape, dtype=torch.int32).cuda()
    weights[-1].view(-1)[flatlocs] += 1
    weights[-1][torch.abs(target_for_occs[-1]) <= truncation] += 3
    weights[-1] = (weights[-1] == 4).float() * (weight_missing_geo - 1) + 1
    factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        weights[h] = weights[h+1][:,:,::2,::2,::2].contiguous()
        factor *= 2
    return weights


def apply_log_transform(sdf):
    sgn = torch.sign(sdf)
    out = torch.log(torch.abs(sdf) + 1)
    out = sgn * out
    return out


def compute_bce_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_loss_masking, truncation=3, batched=True):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    predvalues = sparse_pred_vals.view(-1)
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = tgtvalues != UNK_ID
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues[tgtvalues == UNK_ID] = 0
    if batched:
        loss = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
        else:
            raise
    return loss

def compute_iou_sparse_dense(sparse_pred_locs, dense_tgts, use_loss_masking, truncation=3, batched=True): 
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    corr = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    union = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    for b in range(dense_tgts.shape[0]):
        tgt = dense_tgts[b,0]
        if sparse_pred_locs[b] is None:
            continue
        predlocs = sparse_pred_locs[b]
        # flatten locs # TODO not sure whats the most efficient way to compute this...
        predlocs = predlocs[:,0] * dims[1] * dims[2] + predlocs[:,1] * dims[2] + predlocs[:,2]
        tgtlocs = torch.nonzero(tgt == 1)
        tgtlocs = tgtlocs[:,0] * dims[1] * dims[2] + tgtlocs[:,1] * dims[2] + tgtlocs[:,2]
        if use_loss_masking:
            tgtlocs = tgtlocs.cpu().numpy()
            # mask out from pred
            mask = torch.nonzero(tgt == UNK_ID)
            mask = mask[:,0] * dims[1] * dims[2] + mask[:,1] * dims[2] + mask[:,2]
            predlocs = predlocs.cpu().numpy()
            if mask.shape[0] > 0:
                _, mask, _ = np.intersect1d(predlocs, mask.cpu().numpy(), return_indices=True)
                predlocs = np.delete(predlocs, mask)
        else:
            predlocs = predlocs.cpu().numpy()
            tgtlocs = tgtlocs.cpu().numpy()
        if batched:
            corr += len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union += len(np.union1d(predlocs, tgtlocs))
        else:
            corr[b] = len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union[b] = len(np.union1d(predlocs, tgtlocs))
    if not batched:
        return np.divide(corr, union)
    if union > 0:
        return corr/union
    return -1

def compute_l1_predsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_log_transform, use_loss_masking, known, batched=True, thresh=None):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    locs = sparse_pred_locs if thresh is None else sparse_pred_locs[sparse_pred_vals.view(-1) <= thresh]
    predvalues = sparse_pred_vals.view(-1) if thresh is None else sparse_pred_vals.view(-1)[sparse_pred_vals.view(-1) <= thresh]
    flatlocs = locs[:,3]*dims[0]*dims[1]*dims[2] + locs[:,0]*dims[1]*dims[2] + locs[:,1]*dims[2] + locs[:,2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[flatlocs]
        predvalues = predvalues[mask]
        tgtvalues = tgtvalues[mask]
        if weight is not None:
            weight = weight[mask]
    if use_log_transform:
        predvalues = apply_log_transform(predvalues)
        tgtvalues = apply_log_transform(tgtvalues)
    if batched:
        if weight is not None:
            loss = torch.abs(predvalues - tgtvalues)
            loss = torch.mean(loss * weight)
        else:
            loss = torch.mean(torch.abs(predvalues - tgtvalues))
    else:
        if dense_tgts.shape[0] == 1:
            if weight is not None:
                loss_ = torch.abs(predvalues - tgtvalues)
                loss[0] = torch.mean(loss_ * weight).item()
            else:
                loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise
    return loss

# hierarchical loss 
def compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, truncation, use_log_transform=True, weight_missing_geo=1, input_locs=None, use_loss_masking=True, known=None, batched=True):
    assert(len(output_occs) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    losses = [] if batched else [[] for i in range(len(output_occs) + 1)]
    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation)
    for h in range(len(output_occs)):
        if len(output_occs[h][0]) == 0 or loss_weights[h] == 0:
            if batched:
                losses.append(-1)
            else:
                losses[h].extend([-1] * batch_size)
            continue
        cur_loss_occ = compute_bce_sparse_dense(output_occs[h][0], output_occs[h][1][:,0], target_for_occs[h], weights[h], use_loss_masking, batched=batched)
        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID)*UNK_THRESH
        cur_loss_sdf = compute_l1_predsurf_sparse_dense(output_occs[h][0], output_occs[h][1][:,1], target_for_hier[h], weights[h], use_log_transform, use_loss_masking, cur_known, batched=batched)
        cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            loss += loss_weights[h] * cur_loss
            losses.append(cur_loss.item())
        else:
            loss += loss_weights[h] * cur_loss
            losses[h].extend(cur_loss)
    # loss sdf
    if len(output_sdf[0]) > 0 and loss_weights[-1] > 0:
        cur_loss = compute_l1_predsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, weights[-1], use_log_transform, use_loss_masking, known, batched=batched)
        if batched:
            loss += loss_weights[-1] * cur_loss
            losses.append(cur_loss.item())
        else:
            loss += loss_weights[-1] * cur_loss
            losses[len(output_occs)].extend(cur_loss)
    else:
        if batched:
            losses.append(-1)
        else:
            losses[len(output_occs)].extend([-1] * batch_size)
    return loss, losses

def compute_l1_tgtsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, truncation, use_loss_masking, known, batched=True, thresh=None):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    batch_size = dense_tgts.shape[0]
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    pred_dense = torch.ones(batch_size * dims[0] * dims[1] * dims[2]).to(dense_tgts.device)
    fill_val = -truncation
    pred_dense.fill_(fill_val)
    if thresh is not None:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) <= thresh)
    else:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) < truncation)
    batchids = tgtlocs[:,0]
    tgtlocs = tgtlocs[:,0]*dims[0]*dims[1]*dims[2] + tgtlocs[:,2]*dims[1]*dims[2] + tgtlocs[:,3]*dims[2] + tgtlocs[:,4]
    tgtvalues = dense_tgts.view(-1)[tgtlocs]
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    pred_dense[flatlocs] = sparse_pred_vals.view(-1)
    predvalues = pred_dense[tgtlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[tgtlocs]
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
    if batched:
        loss = torch.mean(torch.abs(predvalues - tgtvalues)).item()
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise
    return loss

