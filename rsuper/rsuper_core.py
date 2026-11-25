"""
rsuper_core.py - R-Super (MICCAI 2025) method on a simple 3D U-Net, without the R-Super repository.

PART 1 is COPIED VERBATIM from github.com/MrGiovanni/R-Super @ 0cac36f (each function is marked with
its file and line range). Nothing in it is edited:
  * rsuper_train/training/losses_foundation.py : the Volume Loss (volume_loss_basic, dice_based_volume_loss),
    the Ball Loss (ball_loss, isolate_tumor, ball_convolution, create_ball_kernel, insert_ball,
    GlobalWeightedRankPooling), the masked segmentation loss (get_known_voxels, DiceLossMultiClass) and
    calculate_loss(), which combines them exactly as in training.
  * rsuper_train/training/augmentation.py : crops (tumour/organ/background crops, crop_foreground_3d used to
    crop report cases on the reported sub-segment), rotation and intensity augmentation.
PART 2 is OUR code: the U-Net (stand-in for MedFormer), the data loader for mask and report cases
(stand-in for dataset_abdomenatlas_UFO.py), the training loop and sliding-window inference.
"""
import os, sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, label

# module-level counters used by the copied debug code (as in losses_foundation.py)
counter = 0
counter2 = 0
counter3 = 0

# #############################################################################################
# PART 1a - COPIED: rsuper_train/training/losses_foundation.py
# #############################################################################################
# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 22-46 (dilate_volume)
def dilate_volume(volume, kernel_size, full_pass_radius=3):
    # ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # for small kernels, just do one pass
    if kernel_size <= (2*full_pass_radius+1):
        return dilate_volume_conv(volume, kernel_size)

    # compute how many "radius‑3" (kernel=7) passes we need
    # radius = (kernel_size‑1)//2  (an integer number of voxels)
    radius = (kernel_size - 1) // 2

    num_full = radius // full_pass_radius  # integer division
    remainder = radius % full_pass_radius  # 0, 1, or 2

    # apply all full radius‑3 passes
    for _ in range(num_full):
        volume = dilate_volume_conv(volume, 2*full_pass_radius + 1)

    # handle the leftover radius if any (1→kernel=3, 2→kernel=5)
    if remainder > 0:
        volume = dilate_volume_conv(volume, 2*remainder + 1)

    return volume


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 50-99 (dilate_volume_conv)
def dilate_volume_conv(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        kernel_size+=1



    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    #we can use a maxpool or a ball convolution to dilate the volume. Maxpool should be faster, but it uses a cube kernel, while the ball kernel is more accurate.
    #dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    ball_kernel = create_ball_kernel(kernel_size).type_as(volume).unsqueeze(0).unsqueeze(0).repeat(volume.shape[1],1, 1, 1, 1)

    # Calculate padding such that the output size is the same as the input size.
    kernel_size = ball_kernel.shape[-1]
    padding = kernel_size // 2

    dilated = F.conv3d(volume, ball_kernel, padding=padding, groups=volume.shape[1])
    #binarize
    dilated = (dilated > 0).float()

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 101-146 (dilate_volume_maxpool)
def dilate_volume_maxpool(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    kernel_size = max(1,int(kernel_size/(2**(0.5))))#compensates for the fact that maxpool is not a round kernel
    if kernel_size%2==0:
        kernel_size+=1

    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number for proper alignment.")

    # Calculate padding such that the output size is the same as the input size.
    padding = kernel_size // 2


    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 150-199 (get_known_voxels)
def get_known_voxels(y: torch.Tensor, unk_voxels: torch.Tensor, dilation=5,sanity=True, classes = None):
    """
    We cannot calculate the BCE loss for voxels we do not know the ground-truth for.
    This function will output a per-voxel masks showing the known voxels. You can use it to mask the loss (or the output and label).
    Args:
        y (torch.Tensor): Tensor of shape (B, C, H, W, D).
        unk_voxels (torch.Tensor): Tensor of shape (B, C, H, W, D) indicating the regions that have tumors not annotated per voxel for each class. I.e., in this tensor, 1 represents voxels we do not know the per-voxel ground-truth. 
        Zero representas voxels we do know the per-voxel ground-truth.
        dilation (int): Size of the cubic structuring element for dilation. Default is 5.
    """
    unk_voxels=unk_voxels.float()
    assert torch.equal(unk_voxels.bool().float(),unk_voxels), 'unk_voxels must be binary'

    if dilation>0:
        #dilate unk voxels: adds a margin around the unknown voxels
        unk_voxels = dilate_volume(unk_voxels, dilation)

    #print("unk_voxels unique values:", torch.unique(unk_voxels), flush=True)
    #print("unk_voxels sum:", unk_voxels.sum(), flush=True)
    one = torch.ones(unk_voxels.shape).type_as(unk_voxels)
    known_voxels = one-unk_voxels
    known_voxels = known_voxels.type_as(y).float()
    assert torch.equal(known_voxels + unk_voxels,one)

    #print('Sum of known voxels:',known_voxels.sum())
    #print('Sum of unknown voxels:',unk_voxels.sum())
    #print('Sum of all voxels:',one.sum(),'matches?',torch.equal(known_voxels + unk_voxels,one))

    if sanity:
        global counter
        if counter<10:
            debug_save_labels(y,str(counter)+'_y',label_names=classes) 
            debug_save_labels(known_voxels,str(counter)+'_known_voxels',label_names=classes)
            debug_save_labels(unk_voxels,str(counter)+'_unk_voxels',label_names=classes)
            print('Saved to '+ str(counter)+'_known_voxels')
            counter+=1



    #print number of channels with unknown voxels
    #num_unknown_channels = unk_voxels.float().sum(dim=(-1,-2,-3))>0
    #num_unknown_channels = num_unknown_channels.float().sum(-1)
    #num_unknown_channels = num_unknown_channels.mean(0)
    #print("---------Number of channels with unknown voxels: ", num_unknown_channels, flush=True, file=sys.stderr)
    #print("Number of known voxels: ", known_voxels.sum(), flush=True, file=sys.stderr)

    #with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
    #    classes = yaml.load(f, Loader=yaml.SafeLoader)

    return known_voxels


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 204-248 (get_lesion_channels)
def get_lesion_channels(out, classes, assertion = False, return_class_names = False):
    #merge lesion channels if they are in the same organ. Outputs will have only lesion channels, removes organ channels.
    assert out.shape[1] == len(classes)
    #print('Shapes here: ', out.shape, chosen_segment_mask.shape, flush=True, file=sys.stderr)

    lesion_out = {}

    for i,clss in enumerate(classes,0):
        #print('Class is:',clss,'Mask sum is:',chosen_segment_mask[:,i].sum())
        for suffix in ['lesion','cyst','pdac','pnet']:
            if suffix in clss:
                name = clss[:clss.index('_'+suffix)+len('_'+suffix)].replace('pancreatic','pancreas')
                if name not in lesion_out:
                    lesion_out[name] = []
                lesion_out[name].append(out[:,i])

    for key in lesion_out.keys():#this combines multi-channel outputs into a single channel
        lesion_out[key] = torch.stack(lesion_out[key],dim=0).max(dim=0).values
        

    #from dicts to tensor
    kys=list(lesion_out.keys())
    lesion_out = torch.stack([lesion_out[key] for key in kys],dim=1).type_as(out)
    
    if assertion:
        for i in range(lesion_out.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = lesion_out[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [kys[j] for j in range(len(kys)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )
    if return_class_names:
        return lesion_out, kys
    else:
        return lesion_out


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 250-349 (volume_loss_basic)
def volume_loss_basic(out,chosen_segment_mask,tumor_volumes, 
                      labels,unk_voxels,
                      classes='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      dilation_segment=31, dilation_unk=7, tolerance=0.1,loss_function='deprecated',n='huber',
                      sigmoid=True, class_weights=None):
    """
    Computes the basic tumor volume loss. This loss compares the total predicted tumor volume inside 
    a subsegment with the tumor volume from the report.
    
    Parameters
    ----------
    out : Tensor
        Network segmentation output. Must match *labels* shape.
    chosen_segment_mask : Tensor
        Same shape; 1 where the report’s lesion lies.
    tumor_volumes : Tensor
        `(B,T)` – *T* ≤ 10; volume of tumors in the chosen_segment_mask region.
    tolerance : float
        Relative dead-zone. 0.1 ⇒ no penalty if |V̂−V| ≤ 10 %·V.
    dilation_segment / dilation_unk : int
        Grow masks to compensate for imperfect sub-segment borders and
        annotation gaps (empirically 31 / 7 gave the most stable
        training).
    class_weights : Optional[Tensor]
        `(B,C)` or `(B,C,1,1,1)`; combats class imbalance.
    sigmoid : bool
        Whether to apply sigmoid activation to the output. Set to True if out are logits.

    Returns
    -------
    dict  {"dice_volume_loss": scalar tensor}
    """
    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    
    if class_weights is not None:
        assert len(class_weights.shape) == 5
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert class_weights.shape[0] == out.shape[0], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        #repeat class weights to match the batch size of out
        class_weights = class_weights.repeat(1, 1, out.shape[2], out.shape[3], out.shape[4]) # B,C,H,W,D
        
    #get only the channels with lesions---apply this in the beginning to reduce computational cost of dilation
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes,assertion=False)
    labels = get_lesion_channels(labels, classes)
    unk_voxels = get_lesion_channels(unk_voxels, classes)

    #activation
    if sigmoid:
        out = torch.sigmoid(out)

    #dilate the chosen segment mask
    chosen_segment_mask = dilate_volume(chosen_segment_mask,dilation_segment)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,dilation_unk)

    #remove from this loss any channel with a tumor that is annotated per-voxel
    per_voxel_positives = (labels.sum((-1,-2,-3),keepdim=True)>0).float()#B,C, which elements we have a tumor annotated per voxel
    #labels = labels * (1-per_voxel_positives)
    out = out * (1-per_voxel_positives)
    
    #voxels we are sure have no tumor:
    negative_voxels = 1 - ((labels + unk_voxels + chosen_segment_mask) > 0).float() #B,C

    
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)
        class_weights = class_weights.mean(dim=(-1,-2,-3)) #reduce to B,C, this will be used to weight the loss for each channel.
    

    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape
    assert out.shape == negative_voxels.shape
    out_in_subsegment = out * chosen_segment_mask
    out_in_negative_voxels = out * negative_voxels

    #we have 1 report volume per batch item, but to what class does it refer to? we can use chosen_segment_mask to figure that out
    report_volume = tumor_volumes.sum(-1) # shape B, we sum the multiple tumors we can have
    report_volume = report_volume.unsqueeze(-1).repeat(1,chosen_segment_mask.shape[1])#B,3
    gate=(chosen_segment_mask.sum(dim=(-1,-2,-3))>0).float()#B,3, one in the lesion channel the report volume refers to, 0 otherwise
    #assert gate.shape[-1]==3
    report_volume = report_volume * gate #B,C, only non-zero for lesion we care about in each CT patch


    loss=dice_based_volume_loss(out_in_subsegment,report_volume,tolerance=tolerance,E=500,cross_entropy=False)
    #shape of loss should be B,C
    if class_weights is not None:
        #apply class weights to the loss
        loss = loss * class_weights
    loss = loss.mean()
    loss={'dice_volume_loss':loss}
    #print('Using dice volume loss')
    assert not torch.isnan(loss['dice_volume_loss']).any(), 'loss is nan'
    return loss


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 352-395 (dice_based_volume_loss)
def dice_based_volume_loss(x,y,tolerance=0.1,E=500,cross_entropy=False):
    """
    This is the function shown in the miccai paper Figure 1.
    """
    #assert no negative values
    assert torch.min(y).item()>=0
    assert torch.min(x).item()>=0

    #assert no nan
    assert not torch.isnan(x).any(), 'Output is nan'
    assert not torch.isnan(y).any(), 'label is nan'

    #tolerance: return 0 if x/y is within 1+/- tolerance
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3))
    assert len(x.shape)==2, f'shape of x is: {x.shape}'

    predicted_volume = x
    target_volume = y

    assert predicted_volume.shape == target_volume.shape

    loss=torch.abs(predicted_volume-target_volume)/(predicted_volume+target_volume+E)
    #E allows this to work when the ground-truth is zero.

    #subtract the loss at tolerance, for continuity
    v=(1-tolerance)*target_volume
    mini = target_volume.clamp(max=100)
    v    = torch.max(v, mini)
    loss_at_tolerance=torch.abs(v-target_volume)/(v+target_volume+E)

    loss=loss-loss_at_tolerance

    #clamp at zero
    loss=torch.clamp(loss,min=0,max=1)

    if cross_entropy:
        #print('Using cross-entropy')
        loss = -torch.log(torch.ones(loss.shape).type_as(loss)-loss+1e-5)
    else:
        #print('Using dice volume without cross-entropy')
        pass

    return loss


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 442-537 (GlobalWeightedRankPooling)
def GlobalWeightedRankPooling(x, N=1000, c=0.75, inverse=False, concentrate=1, return_weights=False,hard_cutoff=False):
    """
    Performs Global Weighted Rank Pooling (GWRP). The weights decay exponentially so that
    the top N voxels receive c% of the total weight.
    Ps: the raw weight at voxel N will be 1-c. 
    So, the inverse weight will be c.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        N (int or torch.Tensor): Number of top voxels to concentrate. If an integer, a scalar
                                 value is used; if a tensor of shape (B, C), each (B,C) pair 
                                 uses its own N.
        c (float): Fraction (e.g. 0.9 for 90%) of the total weight to be concentrated in the top N voxels.
    
    Returns:
        torch.Tensor: The pooled tensor of shape (B, C).
    """
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0).unsqueeze(0)
        reduce=True
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    B, C, H, W, D = x.shape
    L = H * W * D  # total number of voxels per (B, C)
    
    # Sort the spatial elements in descending order.
    x_sorted, sort_indices = torch.sort(x.view(B, C, L), dim=-1, descending=True)
    
    # Compute the decay factor d.
    # If N is a scalar, convert it to a tensor of shape (B, C) with that constant.
    if not torch.is_tensor(N):
        N_tensor = torch.full((B, C), N, dtype=torch.float32, device=x.device)
    else:
        N_tensor = N.to(x.device).float()
    # Ensure N is at least 1.
    N_tensor = torch.clamp(N_tensor, min=1)
    
    # Compute d elementwise: d = (1-c)^(1/N).
    d = (1 - c) ** (1.0 / N_tensor)  # shape (B, C)
    # Reshape d to (B, C, 1) so it can broadcast.
    d = d.unsqueeze(-1)
    
    # Create an index tensor of shape (1, 1, L).
    indices = torch.arange(L, dtype=torch.float32, device=x.device).view(1, 1, L)
    
    # Compute weights: each weight is d^(i), broadcasting over (B, C).
    weights_raw = d ** indices  # shape (B, C, L)
    weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)  # normalize to sum to 1

    #assert that, for a random B,C element, the sum of the first N weights is equal to c
    #rand_b=torch.randint(0,B,(1,))
    #rand_c=torch.randint(0,C,(1,))
    #summed = weights[rand_b, rand_c, :int(N_tensor[rand_b, rand_c].item())].sum()  
    #assert abs(summed.item() - c) < 0.2

    if inverse:
        # For the inverse case we want to ignore the top N voxels.
        # Create a mask that is 0 for indices < N and 1 for indices >= N.
        mask_inv = (indices >= N_tensor.unsqueeze(-1)).float()  # shape (B, C, L)
        # Use the complementary weights for the background: here we use (1 - weights_raw)
        weights = mask_inv * (1 - weights_raw)
        # Note: We do not normalize these weights to sum to 1 because the goal here is to measure
        # the background (i.e. the voxels outside the top N).
    elif concentrate!=1:
        assert concentrate>1, 'concentrate must be greater than 1'
        # Create two masks: one for the top N voxels and one for the rest.
        mask_top = (indices < N_tensor.unsqueeze(-1)).float()      # 1 for indices < N, 0 otherwise
        mask_rest = (indices >= N_tensor.unsqueeze(-1)).float()     # 1 for indices >= N, 0 otherwise
        # Leave top N voxels unchanged and scale the rest by (1/concentrate)
        new_weights = mask_top * weights + mask_rest * (weights / concentrate)
        # Renormalize the weights so they sum to 1.
        weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

    if return_weights:
        if hard_cutoff:
            #make all weights after N zero and re-normalize
            mask_top = (indices < N_tensor.unsqueeze(-1)).float()
            weights = mask_top * weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # We need to return the weights reorganized into the original spatial order.
        # sort_indices tells us, for each (B, C, i), which voxel in the unsorted order that value came from.
        # Compute the inverse permutation.
        inverse_indices = sort_indices.argsort(dim=-1)
        # unsort the weights so that they align with the original order.
        weights_unsorted = weights.gather(dim=-1, index=inverse_indices)
        # Reshape to original spatial dimensions.
        weights_unsorted = weights_unsorted.view(B, C, H, W, D)
        if reduce:
            weights_unsorted = weights_unsorted.squeeze(0).squeeze(0)
        return weights_unsorted
    
    # Compute weighted sum and normalize by the sum of weights.
    pooled = (x_sorted * weights).sum(dim=-1)

    return pooled


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 541-607 (DiceLossMultiClass)
def DiceLossMultiClass(preds, targets, known_voxels, alpha = 0.5, beta=0.5, size_average=True, reduce=True, sigmoid=True, class_weights=None):

    if len(preds.shape)==3:
        preds=preds.unsqueeze(0).unsqueeze(0)
    if len(targets.shape)==3:
        targets=targets.unsqueeze(0).unsqueeze(0)
    if len(known_voxels.shape)==3:
        known_voxels=known_voxels.unsqueeze(0).unsqueeze(0)

    if len(preds.shape)==4:
        preds=preds.unsqueeze(0)
        targets=targets.unsqueeze(0)
        known_voxels=known_voxels.unsqueeze(0)

    assert len(preds.shape)==5
    assert (preds.shape == targets.shape) and (targets.shape == known_voxels.shape), f"Shapes do not match, pred, target and unk are: {preds.shape}, {targets.shape}, {known_voxels.shape}"

    N = preds.size(0)
    C = preds.size(1)
    
    if sigmoid:
        P = torch.sigmoid(preds)
    else:
        P = preds

    P = P * known_voxels
    targets = targets * known_voxels

    smooth = 1e-5

    class_mask = targets

    ones = torch.ones(P.shape).to(P.device)
    P_ = ones - P 
    class_mask_ = ones - class_mask

    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    alpha = alpha.unsqueeze(0).repeat(N, 1) # repeat for each batch item, now alpha is B,C

    alpha = torch.clamp(alpha, min=0.2, max=0.8) 
    #print('alpha:', alpha)
    beta = 1 - alpha
    num = torch.sum(TP, dim=(-1,-2,-3)).float()
    den = num + alpha * torch.sum(FP, dim=(-1,-2,-3)).float() + beta * torch.sum(FN, dim=(-1,-2,-3)).float()

    dice = num / (den + smooth)
    loss = 1 - dice
    if class_weights is not None:
        class_weights = class_weights.mean(dim=(-1,-2,-3))
        while len(class_weights.shape) < len(loss.shape):
            class_weights = class_weights.unsqueeze(0)
        assert class_weights.shape == loss.shape, f'Class weights shape {class_weights.shape} does not match the shape of dice loss {loss.shape}'
        # Apply class weights
        loss = loss * class_weights
    
    if not reduce:
        return loss

    if size_average:
        assert len(loss.shape) == 2, f'Loss should be 2D after reduction, but got {loss.shape}.'
        loss = loss.mean()  # Average over the batch size

    return loss


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 678-682 (merge_no_overlap)
def merge_no_overlap(d1, d2):
    overlap = d1.keys() & d2.keys()
    if overlap:
        raise KeyError(f"Cannot merge: duplicate keys found: {overlap}")
    return {**d1, **d2}


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 685-1076 (calculate_loss)
def calculate_loss(model_output, label, unk_voxels, args, matcher,chosen_segment_mask,
                   tumor_volumes_report,tumor_diameters,
                   classes,input_tensor=None, class_weights=None, model_genesis=False,
                   clip_only=False,report_embeddings=None, dist=None):
    """
    This function calculates all of our loss functions, i.e., the segmentation loss (dice and BCE), and the report supervision
    losses (volume and ball loss), and any baseline loss (clip, classification, models genesis).
    
    Compute *all* losses used by R-Super for a single training step.

    The routine aggregates five possible loss terms
    (segmentation, volume, ball, classification, contrastive) and
    returns them in a dictionary **plus** an `"overall"` key that is the
    weighted sum of all active losses. You can simply back-propagate this 
    overall loss.

    ----------
    Mandatory inputs
    ----------------
    model_output : dict
        What the network returned for the current batch.
        * **segmentation** – required. Either  
          `(B, C, H, W, D)` **or** a *tuple/list* of such tensors when
          deep supervision is enabled.  B is batch size, C is number of classes.
        * **classification** – present only when
          `args.classification_branch` is *True*; shape `(B, Nc)`.  
          This is for the MTL baseline.
        * **clip** – present only when *clip-only* contrastive learning
          is requested. This is for the CLIP-Like baseline.

    label : torch.FloatTensor  
        Voxel-wise one-hot labels with shape `(B, C, H, W, D)` where
        `C == len(classes)`.

    unk_voxels : torch.FloatTensor  
        Same shape as *label*. 1 → the voxel is **unknown** (should be
        *masked-out* from standard segmentation losses), 0 → ground-truth is
        known. Must be binary. unk_voxels marks organs where reports mention
        tumors, but we have no per-voxel annotations for these tumors.
        These are the cases to be optimized by the volume and ball losses.

    args : argparse.Namespace  
        Experiment hyper-parameters and feature flags.  Only the
        following fields are accessed by this pared-down version:  
        `loss`, `aux_weight`, `seg_loss`, `report_volume_loss_basic`,
        `volume_loss_tolerance`, `ball_bce_weight`, `ball_dice_weight`,
        `multi_ch_tumor`, `stardard_ce_ball`.

    matcher : Callable  
        Hungarian matcher used when `args.multi_ch_tumor` is *True*.  
        Signature `out_ids, label_ids = matcher(pred, gt)`. 
        Not used in the MICCAI paper.

    chosen_segment_mask : torch.FloatTensor  
        Binary mask of voxels belonging to the organ sub-segment in
        which tumors resides. Same shape as *label*. This is the organ
        with tumor(s) where our data loader cropped at. We will use the
        volume and ball losses to make the AI find the tumor inside 
        this organ/sub-segment.

    tumor_volumes_report : torch.FloatTensor  
        `(B, T)` where *T* is the max # tumours per crop, set to 10. 
        Each entry is the radiology-report
        volume in voxels. From reports, our data loader calculates volumes in
        mm3. If you use 1x1x1 spacing like we did, this is fine. If not, 
        you must use your voxel spacing to convert mm3 to voxels.

    tumor_diameters : torch.FloatTensor  
        `(B, T, 3)` – three orthogonal diameters (mm) for every tumour.
        Again, we use 1x1x1 spacing. If you use other spacing, you will need
        to convert these diameters to your voxel spacing in the x y axes.
        (using 1x1x1 makes this easier).

    classes : list[str]  
        Alphabetically-sorted class names; length **C**.  Used for
        consistency checks and to locate lesion channels.

    ----------
    Optional
    ------------------------

    input_tensor : torch.Tensor, optional  
        Raw CT patch `(B, 1, H, W, D)` only needed for saving debug
        images when `counter3 < 10`.

    class_weights : torch.FloatTensor, optional  
        Per-sample, per-class weights `(B, C)`.  Will be broadcast to
        `(B, C, H, W, D)` internally.  **Must** be either *None* or a
        true weight map (the code removes it if it equals `torch.ones`).  

    model_genesis : bool, default=False  
        When *True* the function returns a simple voxel-wise MSE loss
        (`model_genesis_loss`) and skips every other path. This is 
        used for the Model Genesis pretraining.

    clip_only : bool, default=False  
        Activates the symmetric InfoNCE contrastive loss between CT
        patch embeddings and report embeddings. Requires
        `model_output['clip']`, *report_embeddings*, and a valid
        `torch.distributed` *dist* object.
        This is used for the CLIP-Like pretraining.

    report_embeddings : torch.Tensor, dist : torch.distributed  
        Only used when *clip_only* is *True*.
        
    -------------
    Important `args` fields still used
    ----------------------------------
    loss (str)               – Selects report-loss type:  
                             ball_dice_last: default, ball (last layer) and volume (deep supervision) losses
                             dice: volume loss only (the name is strange, but it is because we were somewhat inspired by the dice loss to develop this loss)
                             ball: ball loss only

    aux_weight (list[float]) – Per-decoder weight when deep supervision
                            is enabled (one weight per head).

    seg_loss (float)         – Weight for the BCE+Dice
                            segmentation loss.

    report_volume_loss_basic (float)
                            – Weight for all report-driven losses (ball and volume);
                            0 ⇒ those losses are skipped.

    volume_loss_tolerance (float)
                            – ±relative band where the dice-volume loss = 0 (e.g. 0.1 → ±10 %). 
                            Use 0.1 for 10% (what we used).

    ball_bce_weight (float)  – Relative weight of the BCE term inside
                            `ball_loss`. Deafult is 1.

    ball_dice_weight (float) – Relative weight of the Dice term inside
                            `ball_loss`. Deafult is 1.

    multi_ch_tumor (bool)    – Not used in the paper, set to False.

    stardard_ce_ball (bool)  – If True, ball loss uses the “standard”
                            single mean for BCE; otherwise it averages
                            foreground & background separately.

    ----------
    Returned value
    --------------
    dict
        Every active loss under a descriptive key **plus**:
        * `'segmentation'` – BCE-with-logits + Dice (masked)  
        * `'report'` (or the two keys produced by *ball loss*)  
        * `'classification'` (only if `args.classification_branch`)  
        * `'contrastive_loss'` (only in *clip-only* mode)  
        * `'overall'` – sum of all above (or custom wrapper output)
    """
    global counter2
    #print('Unk voxels:', unk_voxels)
    
    if model_genesis:
        return model_genesis_loss(model_output['segmentation'],label)
    
    if clip_only:
        seg_result = model_output['segmentation']
        result_embedding = model_output['clip']
        if isinstance(seg_result, tuple) or isinstance(seg_result, list):
            tmp = 0
            for i in range(len(seg_result)):
                tmp = tmp + seg_result[i].sum()*0
        result_embedding = result_embedding+tmp*0#this is to avoid unused parameter error
        #assert same shape (all dimensions)
        assert result_embedding.shape == report_embeddings.shape, f'Result embedding shape is: {result_embedding.shape}, report embedding shape is: {report_embeddings.shape}'
        loss_ct2rep = nce.info_nce(result_embedding, report_embeddings)
        loss_rep2ct = nce.info_nce(report_embeddings, result_embedding)
        sym_loss = 0.5*(loss_ct2rep + loss_rep2ct)
        sym_loss = sym_loss*dist.get_world_size() #this compensated for the all_gather
        return {'contrastive_loss': sym_loss,
                'overall': sym_loss}
       
    y_class=None
    result = model_output['segmentation']
    if args.classification_branch:
        y_class = model_output['classification']
        #print('Classification shape:', y_class.shape)
    
    if chosen_segment_mask is not None and chosen_segment_mask.sum()>0:
        for b in range(chosen_segment_mask.shape[0]):
            if unk_voxels[b].sum()==0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('unk_voxels should not be all zeros if chosen_segment_mask is not all zeros')
            if tumor_volumes_report[b].sum() == 0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('tumor_volumes_report should not be all zeros if chosen_segment_mask is not all zeros')
    
    #raise ValueError(f'Number of classes in classes: {len(classes)}. Number of classes in label: {label.shape[1]}. Number of classes in result: {result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]}')
    assert len(classes) == label.shape[1], f'Number of classes in classes: {len(classes)} does not match the number of channels in label: {label.shape[1]}'
    assert len(classes) == (result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]), \
    f'Number of classes in result: {(result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1])} does not match the number of channels in label: {label.shape[1]}'
    
    if class_weights is not None and torch.equal(class_weights, torch.ones_like(class_weights)):
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(label.device) #make sure class weights are the same size as cls_out
        assert  class_weights.shape[0] == label.shape[0], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert class_weights.shape[1] == label.shape[1], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert len(class_weights.shape) == 2, f'Class weights should be 2D, but got {class_weights.shape}'
        
    cls_loss = None
    if (y_class is not None):
        cls_loss = classification_loss(y_class, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)

    loss = 0
    loss_report = 0
    loss_segmentation = 0
    if class_weights is not None:
        class_weights = class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        assert len(class_weights.shape)==len(label.shape), f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        
    if isinstance(result, tuple) or isinstance(result, list):
        # if use deep supervision, add all loss together---outpput: [final output, hidden output]
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), f'The unknown region should not cover channels where our label is different from 0---knwon. we got{(known_voxels*label).float().sum()} and {label.float().sum()}'
            known_voxels_original = known_voxels.clone()
            #print('Assertion successful')
        else:
            known_voxels=torch.ones(label.shape).type_as(label)
        for j in range(len(result)):
            r,l = result[j],label
            if j==0 and args.multi_ch_tumor:
                #hungarian algorithm---run on the final output, use same indices on hidden layer outputs
                out_ids, label_ids = matcher(r,l)
            if args.multi_ch_tumor:
                #shuffle accorind to hungarian algo output
                r=r[out_ids]
                l=l[label_ids]
                known_voxels = known_voxels_original[label_ids]
                unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
                chosen_segment_mask = chosen_segment_mask[label_ids]
                assert r.shape == known_voxels.shape, f'Label mismatch, known voxels is: {known_voxels.shape}, r is: {r.shape}'
                assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
            
            #assert no nan in output
            assert not torch.isnan(r).any(), 'Output is nan'
            if args.report_volume_loss_basic > 0:
                if ('ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss) and not (j!=0 and 'last' in args.loss):
                    #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                    #print('Using the ball loss')
                    loss_r = ball_loss (out=r, labels=l, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                        tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                        apply_dice_loss=('dice' in args.loss), input_tensor=input_tensor,
                                        sigmoid=True,
                                        standard_ce=args.stardard_ce_ball, class_weights = class_weights,
                                        single_class= False,
                                        diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin)
                    if 'both' in args.loss:
                        loss_r = merge_no_overlap(loss_r,volume_loss_basic(r, chosen_segment_mask, tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=True, class_weights = class_weights,tolerance=args.volume_loss_tolerance))
                        #print('Both')
                else:
                    loss_r = volume_loss_basic(r, chosen_segment_mask, tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=True, class_weights = class_weights,tolerance=args.volume_loss_tolerance)
                    #print('Using the volume loss')
            else:
                loss_r = torch.tensor(0).type_as(r)


            loss_seg = F.binary_cross_entropy_with_logits(r, l.float(), reduction='none', weight=class_weights)
            
            assert loss_seg.shape == known_voxels.shape, f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            if counter2<5 and j==0:
                label_names = classes
                debug_save_labels(torch.sigmoid(r),str(counter2),out_dir='SanityOutputs',label_names=label_names)
                debug_save_labels(l.float(),str(counter2),out_dir='SanityLabelsBeforeLoss',label_names=label_names)
                debug_save_labels(loss_seg,str(counter2),out_dir='SanityLossBCE',label_names=label_names)
                debug_save_labels(loss_seg*known_voxels,str(counter2),out_dir='SanityLossBCEAfterKnownVoxels',label_names=label_names)
                counter2+=1
            loss_seg = loss_seg * known_voxels
            loss_seg = loss_seg.mean() + DiceLossMultiClass(r, l, known_voxels, sigmoid=True,class_weights=class_weights)
            loss_segmentation = loss_segmentation + args.aux_weight[j] * args.seg_loss * loss_seg

            if not isinstance(loss_r, dict):
                loss_report = loss_report + args.aux_weight[j] * args.report_volume_loss_basic * loss_r
            else:
                if isinstance(loss_report,int):
                    loss_report = {}
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                else:#dict
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        if key not in list(loss_report.keys()):
                            loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                        else:
                            loss_report[key] = loss_report[key] + args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
    else:
        #raise ValueError('Result is not a tuple or list, you should be using deep supervision')
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), 'The unknown region should not cover channels where our label is different from 0---knwon'
        else:
            known_voxels=torch.ones(label.shape).type_as(label)

        if args.multi_ch_tumor:
            out_ids, label_ids = matcher(result,label)
            result=result[out_ids]
            label=label[label_ids]
            assert result.shape == known_voxels.shape
            known_voxels = known_voxels[out_ids]
            unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
            chosen_segment_mask = chosen_segment_mask[label_ids]
            assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
        
        #assert no nan in output
        assert not torch.isnan(result).any(), 'Output is nan'
        if args.report_volume_loss_basic > 0:
            if 'ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss:
                #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                loss_r = ball_loss (out=result, labels=label, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                    tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                    apply_dice_loss=('dice' in args.loss),sigmoid=True,
                                    standard_ce=args.stardard_ce_ball,class_weights=class_weights,
                                    single_class= False,
                                    diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin)
                if 'both' in args.loss:
                    loss_r = merge_no_overlap(loss_r,volume_loss_basic(result,chosen_segment_mask,tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=True, class_weights=class_weights,tolerance=args.volume_loss_tolerance))
                    #print('Both')
            else:
                loss_r = volume_loss_basic(result,chosen_segment_mask,tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=True, class_weights=class_weights,tolerance=args.volume_loss_tolerance)
        else:
            loss_r = torch.tensor(0).type_as(result)


        loss_seg = F.binary_cross_entropy_with_logits(result, label.float(), reduction='none', weight=class_weights) #use BCE with logits for the segmentation loss
    
        assert loss_seg.shape == known_voxels.shape

        loss_seg = loss_seg * known_voxels
        loss_seg = loss_seg.mean() + DiceLossMultiClass(result, label, known_voxels, sigmoid=True,class_weights=class_weights)
        loss_segmentation = loss_segmentation + args.seg_loss * loss_seg
        if not isinstance(loss_r, dict):
            loss_report = loss_report + args.report_volume_loss_basic * loss_r
        else:
            if isinstance(loss_report,int):
                loss_report = {}
            for key in loss_r.keys():
                if key == 'ball_loss_bce':
                    weight = args.ball_bce_weight
                    #print(f'Using the ball bce weight: {weight}')
                elif key == 'ball_loss_dice':
                    weight = args.ball_dice_weight
                    #print(f'Using the ball dice weight: {weight}')
                else:
                    weight = 1
                loss_report[key] = args.report_volume_loss_basic * weight * loss_r[key]
                
    loss={'segmentation':loss_segmentation}
    if isinstance(loss_report,dict):
        for key in loss_report.keys():
            loss[key] = loss_report[key]
    else:
        loss['report'] = loss_report
        
    if cls_loss is not None:
        loss['classification'] = cls_loss

    loss_overall = 0
    for key in loss.keys():
        #print('loss key:', key)
        loss_overall = loss_overall + loss[key]
    

    loss['overall']=loss_overall
    if torch.isnan(loss_overall).any():
        raise ValueError('loss is nan, propagating this can destroy the network weights, STOP!')

    #check if loss_overall requires grad
    assert loss_overall.requires_grad, 'Loss overall should require grad'

    return loss


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1078-1150 (debug_save_labels)
def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./LossChecking",
                      batch_idx = 0):
    """
    Saves each channel of the specified batch index in `labels` as a .nii.gz file.
    
    Args:
        labels (torch.Tensor): A tensor of shape (B, C, H, W, D).
        label_names_yaml (str): Path to a YAML file containing a list of label names.
                                The list will be sorted alphabetically and used
                                to name the channels.
        out_dir (str): Output directory to save the .nii.gz files. Defaults to "LossSanity".
        batch_idx (int): Which batch element to save. Defaults to 0.
    """
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    #raise ValueError(f'Label names is: {label_names}')
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)

    if labels.shape[1]!=len(label_names_sorted):
        raise ValueError(f"Number of channels in labels ({labels.shape[1]}) does not match the number of label names ({len(label_names_sorted)}). Labels loaded from: {label_names}. ")
        label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml'
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)
        label_names_sorted = sorted(label_names)
    
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    if C != len(label_names_sorted):
        label_names_sorted = [str(i) for i in list(range(C))]
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        out_path = os.path.join(out_dir, f"{name}_{channel_label_name}.nii.gz")

        #print(f'Saving: {out_path}, its sum is {channel_data.sum()}')
        
        # Save
        nib.save(nifti_img, out_path)
        
    print(f"Saved to {out_path}")


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1161-1232 (create_ball_kernel)
def create_ball_kernel(diameter, gaussian=False, gaussian_std=1.5):
    """
    Creates a 3D torch tensor (kernel) where there is a 'ball' of a given diameter.
    The diameter is first rounded up to the next odd integer. The kernel size is then
    computed to be 1.2 × (that odd diameter), rounded to the next odd integer.
    
    The ball is centered in this larger kernel. Inside the ball (hard cutoff at the
    ball boundary), values are set to 1 (or to a truncated Gaussian if `gaussian=True`).
    Outside the ball, values are 0. If `gaussian=True`, the Gaussian is centered at
    the ball center with standard deviation `gaussian_std * radius`.

    Parameters
    ----------
    diameter : float or int
        Desired diameter of the ball. Will be rounded up to the next odd integer.
    gaussian : bool, optional
        Whether to fill the ball with a Gaussian distribution, by default False.
    gaussian_std : float, optional
        Standard deviation factor (relative to the ball radius) if gaussian=True.
        For example, if the ball's radius is R and gaussian_std=1.5, the std is
        1.5*R, by default 1.5.

    Returns
    -------
    kernel : torch.FloatTensor
        A 3D tensor of shape (kernel_size, kernel_size, kernel_size) containing
        the ball (or Gaussian ball) centered in the kernel.
    """

    # --- Step 1: Round diameter to next odd integer ---
    diameter_ceil = math.ceil(diameter)
    if diameter_ceil % 2 == 0:
        diameter_ceil += 1
    diameter_odd = diameter_ceil  # The final odd diameter
    
    # --- Step 2: Compute kernel size as 1.2 * diameter_odd, also round up to next odd ---
    kernel_size_float = 1.2 * diameter_odd
    kernel_size_ceil = math.ceil(kernel_size_float)
    if kernel_size_ceil % 2 == 0:
        kernel_size_ceil += 1
    kernel_size = kernel_size_ceil  # The final odd kernel size
    
    # Ball radius (float)
    radius = diameter_odd / 2.0

    # --- Create 1D coordinate grid from 0..(kernel_size-1), shift so center is 0 ---
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords_shifted = coords - center  # center at 0
    
    # --- Compute squared distance (3D) via broadcasting ---
    distance_squared = (coords_shifted[:, None, None] ** 2
                      + coords_shifted[None, :, None] ** 2
                      + coords_shifted[None, None, :] ** 2)
    
    # --- Hard cutoff mask for the ball ---
    mask = (distance_squared <= radius**2).float()
    
    if gaussian:
        # Scale std by the ball's actual radius
        std = gaussian_std * radius
        gaussian_values = torch.exp(-distance_squared / (2.0 * std**2))
        kernel = gaussian_values * mask
        # Normalize so that sum of kernel = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = mask  # Binary ball kernel

    #assert the kernel size is odd
    assert kernel.shape[0] % 2 == 1, f'Kernel size should be odd, got {kernel.shape[0]}'
    
    return kernel


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1271-1299 (ball_convolution)
def ball_convolution(x,diameter,gaussian, gaussian_std):
    """
    Performs a 3D convolution on the input tensor `x` using a ball kernel of diameter `diameter`.
    Optionally, the values inside the ball can follow a Gaussian distribution with standard deviation `gaussian_std`.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
    
    Returns:
        torch.Tensor: Convolved tensor of shape (B, C, H, W, D).
    """
    #if diameter is not odd, add 1:
    if diameter%2==0:
        diameter+=1

    # Create the ball kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    
    # Convert kernel to 5D tensor (B=1, C=1, H, W, D)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform the 3D convolution
    out = F.conv3d(x, kernel, padding=kernel.shape[-1]//2)

    assert out.shape == x.shape, f'Output shape should be the same as input shape, got {out.shape} and {x.shape}'
    return out


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1336-1385 (insert_ball)
def insert_ball(out_spatial, best_center, diameter, margin):
    """
    Places a 'ball' of size diameter * (1 + margin) into out_spatial at the 3D coordinate best_center.
    The 3D ordering is assumed to be (z, y, x).
    """
    # 1) Build the ball kernel for insertion
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)

    # 2) Prepare an empty volume with same shape as out_spatial
    masked_volume = torch.zeros_like(out_spatial)
    
    # 3) Extract shape in (z, y, x) order
    Z, Y, X = masked_volume.shape
    
    # 4) The kernel half-width
    d_half = binary_ball_kernel.shape[-1] // 2
    
    # 5) Unpack best_center as (cz, cy, cx)
    cz, cy, cx = best_center
    
    # 6) Compute overlap in Z dimension
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(Z, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # 7) Compute overlap in Y dimension
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(Y, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # 8) Compute overlap in X dimension
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(X, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # 9) Place the kernel region into masked_volume
    masked_volume[
        vol_z_min:vol_z_max,
        vol_y_min:vol_y_max,
        vol_x_min:vol_x_max
    ] = binary_ball_kernel[
        mask_z_min:mask_z_max,
        mask_y_min:mask_y_max,
        mask_x_min:mask_x_max
    ]

    return masked_volume


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1387-1532 (isolate_tumor)
def isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume,
                  diameter_margin=0.5,volume_margin=0.5):
    """
    Uses a ball convolution over x and applies a maximum operation to find the best
    fitting ball center. Then, it multiplies the input by a volume with the same size
    as the input, but with a binary ball placed at the given object center coordinate.
    Finally, after the multiplication, we find the top N voxels inside the remaining volume.
    N is the tumor volume.

    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball (for convolution).
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): Number of voxels to select as the tumor volume.
    
    Returns:masked_volume should be within 
        torch.Tensor: A binary tumor mask of shape (H, W, D) with 1's in the top N voxels.
    """
    reduce=False
    if len(x.shape)==3:
        reduce=True
        x = x.unsqueeze(0).unsqueeze(0)
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    #round diameter
    diameter = np.round(diameter).astype(int)
    #round tumor volume
    tumor_volume = np.round(tumor_volume).astype(int)

    # Ensure the diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Create the ball kernel for convolution.
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    # Convert kernel to a 5D tensor (shape: 1, 1, H, W, D).
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    #assert volume is not larger than the number of voxels in the ball
    if tumor_volume > 100000:
        assert tumor_volume <= (kernel>0).sum()*1.2, f'Tumor volume should be smaller than the number of voxels in the ball, got {tumor_volume} and {(kernel>0).sum()}'

    if (kernel>0).sum() > tumor_volume:
        #we tolerate numerical erros within a margin of 0.2
        tumor_volume = (kernel>0).sum()-1

    
    # Perform 3D convolution.
    out = F.conv3d(x, kernel, padding=kernel.shape[-1] // 2)

    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape} vs {x.shape}"

    # --- Step 1: Find the best fitting ball center ---
    # Assume x is of shape (1, 1, H, W, D); take the spatial part.
    out_spatial = out[0, 0]  # shape: (H, W, D)
    max_idx = torch.argmax(out_spatial)
    best_center = np.unravel_index(max_idx.item(), out_spatial.shape)  # (cx, cy, cz)
    
    # --- Step 2: Create a binary ball mask at the best center ---
    masked_volume = insert_ball(out_spatial,best_center,diameter,diameter_margin)
    new_dim = diameter
    while masked_volume.sum() < tumor_volume:
        #if the ball is in the border of the image, its volume may be less than the tumor volume, We increase the size of the ball until we reach the tumor volume.
        old_dim = new_dim
        new_dim = int(np.round(new_dim * 1.1))
        print(f'Increasing ball size to {new_dim}, current volume is {masked_volume.sum()}, tumor volume is {tumor_volume}')
        if old_dim == new_dim:
            new_dim += 1
        if new_dim % 2 == 0:
            new_dim += 1
        if new_dim >= max(x.shape[-1], x.shape[-2], x.shape[-3]):
            break
        masked_volume = insert_ball(out_spatial,best_center,new_dim,diameter_margin)
    if tumor_volume < (50**3):
        assert (masked_volume.sum() > tumor_volume*0.5), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'
    if tumor_volume > (6**3):
        assert (masked_volume.sum() < tumor_volume*((1+diameter_margin)**3)*2), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume} and diameter {diameter}'

    # --- Step 3: Multiply the input by the binary ball mask ---
    # x has shape (B, C, H, W, D); expand masked_volume to match.
    #assert no negative value in x
    assert (x >= 0).all(), f'Input tensor should not have negative values, got {x.min()}'
    masked_x = (x * masked_volume.unsqueeze(0).unsqueeze(0))

    # --- Step 4: Find the top N voxels in the masked volume ---
    # Remove batch and channel dimensions.
    masked_x_vol = masked_x[0, 0]
    flattened = masked_x_vol.view(-1)
    # Get indices of the top N voxel values.
    t=min(flattened.shape[-1]-1, tumor_volume)
    margin_small = min(0.5,volume_margin)
    t_small = int(t*(1-margin_small))
    t_small =  max(t_small, min(100,tumor_volume))  # Ensure at 4mm tumor
    t_big = min(flattened.shape[-1]-1,int(tumor_volume*(1+volume_margin)))
    topN_values, topN_indices = torch.topk(flattened, t)
    topN_values_small, topN_indices_small = torch.topk(flattened, t_small)
    topN_values_big, topN_indices_big = torch.topk(flattened, t_big)
    #how many indices? Assert this matches the tumor volume
    assert len(topN_indices) == t, f'Expected {tumor_volume} indices, got {len(topN_indices)}'
    # Create a binary volume: set top N positions to 1, rest to 0.
    tumor_mask_flat = torch.zeros_like(flattened)
    tumor_mask_flat[topN_indices] = 1
    tumor_mask_flat_small = torch.zeros_like(flattened)
    tumor_mask_flat_small[topN_indices_small] = 1
    tumor_mask_flat_big = torch.zeros_like(flattened)
    tumor_mask_flat_big[topN_indices_big] = 1
    
    # Reshape to original spatial dimensions.
    tumor_mask = tumor_mask_flat.view_as(masked_x_vol)
    tumor_mask_small = tumor_mask_flat_small.view_as(masked_x_vol)
    tumor_mask_big = tumor_mask_flat_big.view_as(masked_x_vol)
    # Assert the sum here still matches the tumor volume.
    assert tumor_mask.sum() == t, f'Tumor mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    
    #ensure no tumor_max value is outside the ball
    tumor_mask = tumor_mask * masked_volume
    tumor_mask_small = tumor_mask_small * masked_volume
    tumor_mask_big = tumor_mask_big * masked_volume

    if reduce:
        tumor_mask = tumor_mask.squeeze(0).squeeze(0)

    iters = 0
    while tumor_volume < (50**3) and tumor_mask.sum() < tumor_volume*0.7:
        #zero values inside the ball may not be chosen as the top N voxels. In such cases, we dilate the mask
        print(f'dilating tumor mask, iteration {iters}, current volume is {tumor_mask.sum()}, tumor volume is {tumor_volume}')
        if iters >5:
            return tumor_mask, tumor_mask_small, tumor_mask_big
        #dilate the mask
        tumor_mask = dilate_volume(tumor_mask, 7)*masked_volume
        tumor_mask_small = dilate_volume(tumor_mask_small, 7)*masked_volume
        tumor_mask_big = dilate_volume(tumor_mask_big, 7)*masked_volume
        iters+=1

    if tumor_volume < (50**3):
        assert (tumor_mask.sum() > tumor_volume*0.5), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'
    if tumor_volume > (5**3):
        assert (tumor_mask.sum() < tumor_volume*((1+volume_margin)**3)*3), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    #assert it is binary
    assert (tumor_mask == 0).sum() + (tumor_mask == 1).sum() == tumor_mask.numel(), f'Tumor mask should be binary, got {tumor_mask.sum()}'

    return tumor_mask, tumor_mask_small, tumor_mask_big


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1537-1864 (ball_loss)
def ball_loss(out, labels, unk_voxels, chosen_segment_mask, tumor_volumes, tumor_diameters, classes, apply_dice_loss,
              diameter_margin=0.2, volume_margin=0.2, gaussian=True, 
              gaussian_std=1.5, gwrp=True, gwrp_concentration=0.5, dilation_for_background=7,
              subseg_dilation=31,input_tensor=None, unk_dilation=1,
              sigmoid=True, standard_ce=False, class_weights=None,
              single_class=False, use_small_pseudo_mask=True):
    """
    Implementation of the Ball Loss. It uses ball convolutions to localize the tumor, then turns that localization into a voxel-wise training signal. 
    
    Important: this funciton only supervises the segmentation channels for one tumor type per sample (corresponding to the organ we cropped on, encoded in chosen_segment_mask).
    All other tumor channels must still be trained to predict no tumor in case the report mentions they have no tumor, or they must not be penalized in case organs have tumors but are only partially in the crop.
    We handle this with a standard loss (cross-entropy with target 0) on the no-tumor channels. 
    In our calculate_loss function, this is implemented by matching the model outputs to the labels everywhere except at unk_voxels, which are excluded from supervision (loss_seg = loss_seg * known_voxels).
    
    This funciton first uses a ball loss to isolate the tumor. Then, it selects the top N voxels inside the ball as a "pseudo-label" and applies BCE loss per-voxel.
    Important note: this loss is desgined to optimize the segmentation channels where the report mention tumors. The channels without tumor must also be penalize. 
    This can be done with a standard cross-entropy loss with target 0 over these channels. For example, if a report mentions spleen tumors, this loss will penalize the
    spleen tumor channel, another loss (cross-entropy) should penalize the channel for esophagus tumors, bladder tumors,... We do this in our calculate_loss function,
    where the model output output is penalized to match labels, except for the unk_voxels. 
    Args:
    x is the model output
    tumor_diameter is a tensor of size B,T,3, batch, number of tumors in the crop, and 3 diameters
    diameter_margin: how much much we want the ball diameter to be bigger than the maximum tumor diameter
    gaussian: if a gaussian kernel is used in the ball convolution for better centering on the tumor
    gaussian_std: the higher, the smaller the difference between the ball kernel center and border values
    gwrp: wether to use GRWP to average each BCE loss. If so, more weight is given to increasing high confidence voxels.
    sigmoid: wether to apply sigmoid to the output.
    dilation_for_background: we apply a dilation kernel of this size to the tumor pseudo-mask, and define everything outside this mask as background, and use BCE loss to make the backgropund 0
    subseg_dilation: how much we dilate the tumor subsegment. Radiologists/AI may not be super precise when defining the subsegment, and tumors may grow out of organs, so we add a generous margin here.
    standard_ce: if True, we use a standard averaging for the BCE loss. Otherwise, we acerage the foreground and background voxel losses separately, and then sum the two losses.
    class_weights: optional 5D tensor to apply class weights. This is useful when dealing with imbalanced positives and negatives per class or datasets with many classes.
    Important: this loss assumes the output resolution is 1x1x1 mm, and that diammeters are in mm and volumes in mm^3. If the resolution is different, you should adjust the diameters and volumes accordingly or introduce a scaling factor.
    """
    global counter3

    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    if class_weights is not None:
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert len(class_weights.shape) == 5, f'Class weights should be 5D tensor, got {class_weights.shape}'
        #repeat channels to match the output shape
        class_weights = class_weights.repeat(out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4])


    #get only the channels with lesions
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes, assertion=False)
    unk_voxels = get_lesion_channels(unk_voxels, classes)
    labels = get_lesion_channels(labels, classes)
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)

    chosen_segment_mask = dilate_volume(chosen_segment_mask,subseg_dilation)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,unk_dilation)
    to_penalize = torch.ones_like(out)
    #remove the unk voxels from the penalization
    to_penalize = to_penalize * (1 - unk_voxels)
    #also remove the knwon labels
    to_penalize = to_penalize * (1 - labels)
    #but add back the chosen segment mask
    to_penalize = to_penalize + chosen_segment_mask
    #binarize
    to_penalize = (to_penalize > 0).float()


    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape

    losses = []
    losses_dice = []

    for B in range(out.shape[0]):#batch itens
        #assert diameters and violumes make sense
        assert torch.equal(tumor_diameters[B].sum(-1)>0, tumor_volumes[B]>0), f'Tumor diameters and volumes should be consistent, got {tumor_diameters[B]} and {tumor_volumes[B]}'
        
        #get correct batch and class
        x = out[B]
        tumor_seg = chosen_segment_mask[B]
        #current_x is still 4 D, with one class per tumor type. Assert at most one of these channels is non-zero (due to the chosen_segment_mask):
        assert (tumor_seg.sum((-1,-2,-3))>0).float().sum()<=1, f'Only one channel should be non-zero, got {tumor_seg.sum((-1,-2,-3))}'
        
        # if no tumor in this batch, create a zero pseudo label
        if tumor_seg.sum()==0 or tumor_volumes[B].sum()==0:
            # no tumor in this batch, create a zero pseudo label
            pseudo_mask = torch.zeros_like(x)
            if sigmoid:
                if not single_class:
                    #standard, use sigmoid
                    loss = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                else:
                    #use softmax
                    loss = F.cross_entropy(x, pseudo_mask, reduction='none')
                #print('ball loss uses BCE with logits')
            else:
                if not single_class:
                    #assert x is in the range 0-1
                    assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                    #assert pseudo_mask is in the range 0-1
                    assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                    loss = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                else:
                    #single class, but consider that softmax was already applied. Thus, use nll loss
                    #from one-hot to class indices: argmax
                    loss = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
            assert loss.shape == tumor_seg.shape
            loss = loss * to_penalize[B]
            if class_weights is not None:
                # apply class weights if provided
                loss = loss * class_weights[B]
            loss = loss.mean()
            if apply_dice_loss:
                if class_weights is not None:
                    w = class_weights[B]
                else:
                    w = None
                dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                losses_dice.append(dice_loss)
            losses.append(loss.mean())
            continue
        
        #get tumor class
        for c in range(x.shape[0]):
            if tumor_seg[c].sum()>0:
                x = x[c]
                penalize = to_penalize[B][c]
                if class_weights is not None:
                    c_weight = class_weights[B][c] #get the class weights for this batch and class
                else:
                    c_weight = None
                break
        tumor_seg = tumor_seg.sum(0)
        current_tumor_diameters = tumor_diameters[B]
        current_tumor_volumes = tumor_volumes[B]

        # Get the sort indices for tumor_volumes in descending order
        sorted_indices = torch.argsort(current_tumor_volumes, descending=True)

        # Filter indices to keep only those with volume > 0
        sorted_indices = sorted_indices[current_tumor_volumes[sorted_indices] > 0]
        #print('--------Sorted indices:', sorted_indices)
        #print('--------SORTED VOLUMES:', current_tumor_volumes[sorted_indices])
        #print('--------UNSORTED VOLUMES:', current_tumor_volumes)

        #Create the pseudo-mask
        pseudo_masks = []
        pseudo_masks_small = []
        pseudo_masks_big = []
        #update x for the next tumor: remove pseudo_mask, so that this tumor is not selected again.
        if sigmoid:
            x_iter = torch.sigmoid(x)*tumor_seg
        else:
            x_iter = x*tumor_seg
        for tumor_idx in sorted_indices:
            vol=current_tumor_volumes[tumor_idx].item()
            dia=current_tumor_diameters[tumor_idx]
            #get the maximum diameter
            max_diameter = torch.max(dia).item()
            assert max_diameter>0, f'Tumor diameter should be larger than 0, got {max_diameter}'
            assert vol>0, f'Tumor volume should be larger than 0, got {vol}'
            if vol==0 or max_diameter == 0:
                print('Found 0 tumor where it should not be')
                continue
            if max_diameter <= 1:
                print('Found 1mm diameter, increasing to 3')
                max_diameter = 3
            if vol <= 1:
                print('Found 1mm volume, increasing to 9')
                vol = 9
            #assert it is not zero
            #ball convolution: use isolate_tumor to get the top 'tumor_volume' voxels in the outpus, inside the best fitting ball position
            pseudo_mask,pseudo_mask_small,pseudo_mask_big = isolate_tumor(x_iter, diameter=max_diameter, 
                                                                          gaussian=gaussian, gaussian_std=gaussian_std, tumor_volume=vol,
                                                                          diameter_margin=diameter_margin,volume_margin=volume_margin)
            pseudo_masks.append(pseudo_mask)
            pseudo_masks_small.append(pseudo_mask_small)
            pseudo_masks_big.append(pseudo_mask_big)
            x_iter = x_iter * (1 - pseudo_mask) #remove the pseudo mask from the output, so that it is not selected again
        #stack the pseudo masks
        if use_small_pseudo_mask:
            pseudo_mask = torch.stack(pseudo_masks_small).sum(0)
        else:
            pseudo_mask = torch.stack(pseudo_masks).sum(0)
        pseudo_mask = (pseudo_mask > 0).float()
        dilated_pseudo_mask = torch.stack(pseudo_masks_big).sum(0)
        dilated_pseudo_mask = (dilated_pseudo_mask > 0).float()

        #we can add a tolerance margin around the pseudo mask, where we do not penalize the outputs for not being zero
        if dilation_for_background>0:
            dilated_pseudo_mask=dilate_volume(dilated_pseudo_mask, dilation_for_background)
            
        border = dilated_pseudo_mask - pseudo_mask
        #threshold at 0
        border = (border > 0).float()
            
        penalize=penalize * (1 - border)
        #penalize is a tensor with the voxels where we want to apply our losses to here

        #BCE loss with mask
        if sigmoid:
            if not single_class:
                BCE = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
            else:
                #single class
                BCE = F.cross_entropy(x, pseudo_mask, reduction='none')
        else:
            if not single_class:
                #assert x is in the range 0-1
                assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                #assert pseudo_mask is in the range 0-1
                assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                BCE = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
            else:
                #single class, but consider that softmax was already applied. Thus, use nll loss
                #from one-hot to class indices: argmax
                BCE = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
        assert (penalize.shape==BCE.shape), f'To penalize and BCE should have the same shape, got {penalize.shape} and {BCE.shape}'
        BCE = BCE * penalize #cut the loss gradient in the border. Remember that unk voxels were already removed from x

        #dice loss
        #dice loss
        if apply_dice_loss:
            #remove tumor surroundings, to avoid penalizing them: we are not super sure if this region is tumor or not.
            dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
            if sigmoid:
                print('Dice loss:',dice_loss, 'Mean prediction:',torch.sigmoid(x).mean())
            else:
                print('Dice loss:',dice_loss, 'Mean prediction:',x.mean())
            #we make all voxels knwon because we alreay removed unknown voxels from x
            #print('Using dice loss inside the ball loss')

        if not standard_ce:
            #we separate foreground and background, calculate the average per-voxel loss for them separatelly, than sum it. We can use GRWP in the foreg. or not.
            if gwrp:
                #we do BCE for the entire channel, but we do not simply average it. We can use GWRP to average the tumor values (positive GT)
                #we add the pseudo-mask to boost its voxels values and concentrate GWRP there.
                assert pseudo_mask.sum() > 0, f'Pseudo mask should have at least one voxel, got {pseudo_mask.sum()}, volume is {vol} and diameter is {max_diameter}'
                if sigmoid:
                    foreg_weights = GlobalWeightedRankPooling(torch.sigmoid(x)*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                else:
                    foreg_weights = GlobalWeightedRankPooling(x*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                #print highest and lowest non-zero values in foreg_weights
                assert foreg_weights.sum() > 0.95 and foreg_weights.sum() < 1.05, f'GWRP weights should be normalized to 1, got {foreg_weights.sum()}'
                #renormlize gwrp weights so they sum to pseudo_mask.sum()
                foreg_weights = foreg_weights * pseudo_mask.sum()
                #print('GWRP Foreg weights range:', foreg_weights[foreg_weights>0].max(), foreg_weights[foreg_weights>0].min())
                #assert sum of foreg_weights is close to 1
                foreg_weights = foreg_weights*pseudo_mask
                assert BCE.shape == foreg_weights.shape, f'BCE and GWRP weights should have the same shape, got {BCE.shape} and {foreg_weights.shape}'
                loss_foreground = (BCE*foreg_weights)#.mean() #we can use mean here because 
            else:
                #print('Using simple mean for BCE loss')
                loss_foreground = (BCE*pseudo_mask)#.mean()
            
            #Background:
            bkg_weights = 1 - dilated_pseudo_mask
            loss_background = (BCE*bkg_weights)#.mean()
            
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(loss_background.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == loss_background.shape[0], f'Class weights {class_weights[B].shape} do not match loss_background shape {loss_background.shape}'
                loss_foreground = loss_foreground * c_weight
                loss_background = loss_background * c_weight
            loss_foreground = loss_foreground.mean()
            loss_background = loss_background.mean()

            loss = loss_foreground + loss_background
            losses.append(loss)#BCE loss
        else:
            #print('Using standard CE for BCE loss')
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(BCE.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == BCE.shape[0], f'Class weights {c_weight.shape} do not match BCE shape {BCE.shape}'
                BCE = BCE * c_weight
            BCE = BCE.mean()
            losses.append(BCE)#simple mean.

        if apply_dice_loss:
            losses_dice.append(dice_loss.mean())

        if counter3<10:

            counter3+=1
            os.makedirs('SanityBallLoss/'+str(counter3), exist_ok=True)
            if sigmoid:
                save_tensor_as_nifti(torch.sigmoid(x),'SanityBallLoss/'+str(counter3)+'/x')
            else:
                save_tensor_as_nifti(x,'SanityBallLoss/'+str(counter3)+'/x')
            save_tensor_as_nifti(pseudo_mask,'SanityBallLoss/'+str(counter3)+'/pseudo_mask')
            save_tensor_as_nifti(border,'SanityBallLoss/'+str(counter3)+'/border')
            save_tensor_as_nifti(tumor_seg,'SanityBallLoss/'+str(counter3)+'/tumor_segment')
            save_tensor_as_nifti((to_penalize[B].sum(0)>0).float(),'SanityBallLoss/'+str(counter3)+'/to_penalize')
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[B].squeeze(),'SanityBallLoss/'+str(counter3)+'/input_volume')

            #save tumor volumes and diameters as yaml
            with open('SanityBallLoss/'+str(counter3)+'/tumor_volumes.yaml', 'w') as file:
                yaml.dump(tumor_volumes.tolist(), file)
            with open('SanityBallLoss/'+str(counter3)+'/tumor_diameters.yaml', 'w') as file:
                yaml.dump(tumor_diameters.tolist(), file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/known_voxels')
            l=losses[-1].item()
            if apply_dice_loss:
                l+=losses_dice[-1].item()
            if sigmoid:
                info=f'Volume in output: {torch.sigmoid(x).sum().item()}, Volume in report: {vol}, Loss: {l}'
            else:
                info=f'Volume in output: {x.sum().item()}, Volume in report: {vol}, Loss: {l}'
            print(info)
            #save the loss as yaml
            with open('SanityBallLoss/'+str(counter3)+'/loss.yaml', 'w') as file:
                yaml.dump(l, file)
            #save the info as yaml
            with open('SanityBallLoss/'+str(counter3)+'/info.yaml', 'w') as file:
                yaml.dump(info, file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/loss.yaml')

    return {'ball_loss_bce':torch.stack(losses).mean(),
            'ball_loss_dice':torch.stack(losses_dice).mean() if apply_dice_loss else torch.zeros_like(torch.stack(losses).mean())}


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/losses_foundation.py lines 1867-1893 (save_tensor_as_nifti)
def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
    if 'nii.gz' not in filename:
        filename += '.nii.gz'
        
    assert len(tensor.squeeze(0).shape)==3, f"Input tensor should be 3D, got {tensor.shape}"

    # Ensure tensor is on CPU and convert to numpy array.
    np_array = tensor.detach().cpu().numpy()
    
    # If the tensor has an extra channel dimension, squeeze it.
    if np_array.ndim == 4 and np_array.shape[0] == 1:
        np_array = np_array.squeeze(0)
    
    # Create an identity affine (voxel sizes = 1 mm in all directions).
    affine = np.eye(4)
    
    # Create the NIfTI image and save.
    nifti_img = nib.Nifti1Image(np_array, affine)
    nib.save(nifti_img, filename)
    print(f"Saved NIfTI file to {filename}")


# #############################################################################################
# PART 1b - COPIED: rsuper_train/training/augmentation.py
# #############################################################################################
# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 17-19 (gaussian_noise)
def gaussian_noise(tensor_img, std, mean=0):
    
    return tensor_img + torch.randn(tensor_img.shape).to(tensor_img.device) * std + mean


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 34-46 (generate_3d_gaussian_kernel)
def generate_3d_gaussian_kernel(kernel_size, sigma):
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    z = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(x, y, z)

    # Calculate the 3D Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2) ** 1.5
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 48-66 (gaussian_blur)
def gaussian_blur(tensor_img, sigma_range=[0.5, 1.0]):

    sigma = torch.rand(1) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        kernel = generate_3d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(3)]

        return F.conv3d(tensor_img, kernel, padding=padding)
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        kernel = generate_2d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(2)]

        return F.conv2d(tensor_img, kernel, padding=padding)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 69-83 (brightness_additive)
def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    
    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(tensor_img.device)
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(tensor_img.device)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img + rand_brightness


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 86-103 (brightness_multiply)
def brightness_multiply(tensor_img, multiply_range=[0.7, 1.3], per_channel=False):

    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    assert multiply_range[1] > multiply_range[0], 'Invalid range'

    span = multiply_range[1] - multiply_range[0]
    if len(tensor_img.shape) == 5:
        rand_brightness = torch.rand(size=(1, C, 1, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.rand(size=(1, C, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img * rand_brightness


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 106-138 (gamma)
def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=True):
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        _, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        _, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')
    
    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism

    rng = maxm - minm

    mean = tensor_img.mean(dim=1).unsqueeze(1)
    std = tensor_img.std(dim=1).unsqueeze(1)
    gamma = torch.rand(C, 1).to(tensor_img.device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

    if retain_stats:
        tensor_img -= tensor_img.mean(dim=1).unsqueeze(1)
        tensor_img = tensor_img / tensor_img.std(dim=1).unsqueeze(1) * std + mean

    if dim == '3d':
        return tensor_img.reshape(1, C, D, H, W)
    else:
        return tensor_img.reshape(1, C, H, W)


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 140-169 (contrast)
def contrast(tensor_img, contrast_range=(0.65, 1.5), per_channel=False, preserve_range=True):

    if len(tensor_img.shape) == 5:
        dim = '3d'
        _, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        _, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism


    mean = tensor_img.mean(dim=1).unsqueeze(1)
    factor = torch.rand(C, 1).to(tensor_img.device) * (contrast_range[1] - contrast_range[0]) + contrast_range[0]

    tensor_img = (tensor_img - mean) * factor + mean

    if preserve_range:
        tensor_img = torch.clamp(tensor_img, min=minm, max=maxm)

    if dim == '3d':
        return tensor_img.reshape(1, C, D, H, W)
    else:
        return tensor_img.reshape(1, C, H, W)


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 228-319 (random_scale_rotate_translate_3d)
def random_scale_rotate_translate_3d(tensor_img, tensor_lab, scale=0.3, rotate=45, translate=0.1, shear=0.05, foreground=None):
    '''
    The axis order of SimpleITK is x,y,z
    The axis order of numpy/tensor is z,y,x
    The arguments of all transformation should use the numpy/tensor order: [z,y,x]

    '''
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 3

    scale_x = np.random.uniform(low=1-scale[0], high=1/(1-scale[0]))
    scale_y = np.random.uniform(low=1-scale[1], high=1/(1-scale[1]))
    scale_z = np.random.uniform(low=1-scale[2], high=1/(1-scale[2]))

    shear_xy = np.random.uniform(-shear[0], shear[0]) # contribution of y index to x axis
    shear_xz = np.random.uniform(-shear[0], shear[0]) # contribution of z index to x axis
    shear_yx = np.random.uniform(-shear[1], shear[1]) # contribution of x index to y axis
    shear_yz = np.random.uniform(-shear[1], shear[1]) # contribution of z index to y axis
    shear_zx = np.random.uniform(-shear[2], shear[2]) # contribution of x index to z axis
    shear_zy = np.random.uniform(-shear[2], shear[2]) # contribution of y index to z axis

    translate_x = np.random.uniform(-translate[0], translate[0])
    translate_y = np.random.uniform(-translate[1], translate[1])
    translate_z = np.random.uniform(-translate[2], translate[2])


    theta_scale = torch.tensor([[scale_x, shear_xy, shear_xz, translate_x],
                                [shear_yx, scale_y, shear_yz, translate_y],
                                [shear_zx, shear_zy, scale_z, translate_z], 
                                [0, 0, 0, 1]]).float()
    angle_x = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi 
    # rotate along x axis (x index fix, rotae in yz plane)
    angle_y = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
    # rotate along y axis (y index fix, rotate in xz plane)
    angle_z = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    # rotate along z axis (z index fix, rotate in xy plane)
    
    theta_rotate_x = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(angle_x), -math.sin(angle_x), 0],
                                    [0, math.sin(angle_x), math.cos(angle_x), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_y = torch.tensor([[math.cos(angle_y), 0, -math.sin(angle_y), 0],
                                    [0, 1, 0, 0],
                                    [math.sin(angle_y), 0, math.cos(angle_y), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_z = torch.tensor([[math.cos(angle_z), -math.sin(angle_z), 0, 0],
                                    [math.sin(angle_z), math.cos(angle_z), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).float()

    theta = torch.mm(theta_rotate_x, theta_rotate_y)
    theta = torch.mm(theta, theta_rotate_z)
    
    theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    assert len(tensor_img.size())==5
    grid = F.affine_grid(theta, tensor_img.size(), align_corners=True).to(tensor_img.device)
    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
    
    # --- random_scale_rotate_translate_3d ---
    if foreground is not None:
        add_chan = False
        if foreground.ndim == 3:            # (D,H,W)
            foreground = foreground.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            add_chan = 2
        elif foreground.ndim == 4:
            foreground = foreground.unsqueeze(0)  # (1,1,D,H,W)
            add_chan = 1
        elif foreground.ndim == 5:
            add_chan = 0
        else:
            raise ValueError('Invalid dimension of foreground mask')
            
        foreground = F.grid_sample(
            foreground.float(), grid,
            mode='nearest', padding_mode='zeros',
            align_corners=True)
        if add_chan==2:
            foreground = foreground[0,0]    # back to (D,H,W)
        if add_chan==1:
            foreground = foreground[0]
        foreground = foreground.bool()
        return tensor_img, tensor_lab, foreground
    else:
        return tensor_img, tensor_lab


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 446-469 (crop_3d)
def crop_3d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 498-559 (crop_around_coordinate_3d)
def crop_around_coordinate_3d(tensor_img, tensor_lab, crop_size, coordinate, mode, foreground=None):
    assert mode in ['random', 'center', 'small_rnd_shift'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    z, y, x = coordinate

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    
    if mode == 'random':
        min_z = max(0, z-crop_size[0])
        max_z = min(diff_D, z+crop_size[0])
        min_y = max(0, y-crop_size[1])
        max_y = min(diff_H, y+crop_size[1])
        min_x = max(0, x-crop_size[2])
        max_x = min(diff_W, x+crop_size[2])
        
        rand_z = np.random.randint(min_z, max_z)
        rand_y = np.random.randint(min_y, max_y)
        rand_x = np.random.randint(min_x, max_x)
    elif mode == 'small_rnd_shift':
        # Calculate centered crop start indices
        center_z = z - crop_size[0] // 2
        center_y = y - crop_size[1] // 2
        center_x = x - crop_size[2] // 2

        # Maximum random shift (50% of crop size)
        max_shift_z = int(crop_size[0] * 0.5)
        max_shift_y = int(crop_size[1] * 0.5)
        max_shift_x = int(crop_size[2] * 0.5)

        # Generate random offsets in the range [-max_shift, max_shift]
        offset_z = np.random.randint(-max_shift_z, max_shift_z + 1)
        offset_y = np.random.randint(-max_shift_y, max_shift_y + 1)
        offset_x = np.random.randint(-max_shift_x, max_shift_x + 1)

        # Apply offsets and clip to image boundaries
        rand_z = np.clip(center_z + offset_z, 0, D - crop_size[0])
        rand_y = np.clip(center_y + offset_y, 0, H - crop_size[1])
        rand_x = np.clip(center_x + offset_x, 0, W - crop_size[2])
    else:
        min_z = max(0, z - math.ceil(crop_size[0] / 2))
        rand_z = min(min_z, D - crop_size[0])
        min_y = max(0, y - math.ceil(crop_size[1] / 2))
        rand_y = min(min_y, H - crop_size[1])
        min_x = max(0, x - math.ceil(crop_size[2] / 2))
        rand_x = min(min_x, W - crop_size[2])

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    if foreground is not None:
        while len(foreground.shape) < len(tensor_img.shape):
            foreground = foreground.unsqueeze(0)
        cropped_foreground = foreground[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
        return (cropped_img.contiguous(), cropped_lab.contiguous(), cropped_foreground.contiguous())
    else:
        return cropped_img.contiguous(), cropped_lab.contiguous()


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 600-660 (random_crop_on_tumor)
def random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, d, h, w, tumor_case,
                         tumor_prob=None, foreground_prob=None, background_prob=None,
                         return_crop_organ=False, class_names=None,
                         foreground_classes=None):
    """
    Perform a cropping operation on a tensor, based on tumor presence or other criteria.

    Parameters:
        tensor_img (torch.Tensor): The input image tensor.
        tensor_lab (torch.Tensor): The label tensor.
        lesion_classes (list or torch.Tensor): Indices of lesion classes in the label tensor.
        d, h, w (int): Dimensions for cropping.

    Returns:
        tensor_img (torch.Tensor): The cropped image tensor.
        tensor_lab (torch.Tensor): The cropped label tensor.

    """
    rnd = np.random.random()
    if (tumor_prob is None) or (foreground_prob is None) or (background_prob is None):
        #standard probs
        if tumor_case:
            tumor_prob=0.9
            foreground_prob=0.05
            background_prob=0.05
            print('Tumor case')
        else:
            tumor_prob=0
            foreground_prob=0.9
            background_prob=0.1
            print('Non-tumor case')

    if rnd < tumor_prob:
        print('Attempting tumor crop')
        t = tumor_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=return_crop_organ)
        if return_crop_organ:
            #print(f'Tumor crop successful, organ: {crop_organ}')
            tensor_img, tensor_lab, crop_organ = t
        else:
            #print(f'Tumor crop successful, organ: {crop_organ}')
            tensor_img, tensor_lab = t
    elif rnd < (tumor_prob + background_prob):
        print('Background crop')
        tensor_img, tensor_lab = negative_crop(tensor_img, tensor_lab, lesion_classes, d, h, w)
        crop_organ = 'random'
    else:
        print('Attempting organ crop')
        t = organ_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=return_crop_organ,
                       foreground_classes=foreground_classes)
        if return_crop_organ:
            tensor_img, tensor_lab, crop_organ = t
        else:
            tensor_img, tensor_lab = t
    if return_crop_organ:
        if (crop_organ is not None) and crop_organ != 'random':
            crop_organ = class_names[crop_organ]
        print(f'Organ crop successful, organ: {crop_organ}')
        return tensor_img, tensor_lab, crop_organ
    else:
        #print(f'Organ crop successful, organ: {crop_organ}')
        return tensor_img, tensor_lab


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 662-673 (negative_crop)
def negative_crop (tensor_img, tensor_lab, lesion_classes, d, h, w):
    # Negative crop
    foreground_mask = tensor_lab[0].sum(0, keepdim=False)
    back_voxels = torch.nonzero(foreground_mask == 0, as_tuple=False)
    if len(back_voxels) == 0:
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
    else:
        # Crop around a random background voxel
        center = back_voxels[torch.randint(0, len(back_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    return tensor_img, tensor_lab


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 675-714 (organ_crop)
def organ_crop(tensor_img, tensor_lab, lesion_classes, d, h, w,
               return_crop_organ=False,foreground_classes=None):
    # Random label crop
    foreground_mask = tensor_lab[0]#.sum(0, keepdim=False)
    foreg = []
    clss = []
    print('Foreground classes:', foreground_classes)
    for c in range(tensor_lab.shape[1]): # Loop through each class in the label tensor
        if c not in lesion_classes:
            if foreground_classes is not None:
                if c not in foreground_classes:
                    continue
            # Only consider non lesion classes for foreground
            fore = tensor_lab[0][c]
            if fore.sum() > 0:
                foreg.append(fore)
                clss.append(c)
    if len(foreg) !=0:
        #randomly choose one of the foreground classes to crop around
        idx = torch.randint(0, len(foreg), (1,)).item() # randomly choose one of the foreground classes
        foreground_mask = foreg[idx] # choose the randomly selected foreground mask
        crop_organ = clss[idx]
        print('Cropping on organ (not tumor):', crop_organ)
        foreground_voxels = torch.nonzero(foreground_mask)
    else:
        print('Foreground voxels empty, falling back to random crop')
        foreground_voxels = []
        crop_organ = 'random'
    if len(foreground_voxels) == 0:
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        crop_organ = 'random'
    else:
        # Crop around a random foreground voxel
        center = foreground_voxels[torch.randint(0, len(foreground_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    if return_crop_organ:
        return tensor_img, tensor_lab, crop_organ
    else:
        return tensor_img, tensor_lab


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 716-742 (tumor_crop)
def tumor_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=False):
    # Tumor crop
    #print('Applying Tumor crop')
    tumor_mask = tensor_lab[0][lesion_classes]#.sum(1, keepdim=False)---do not sum, this will favor larger tumors. Choose the tumor class randomly across the classes with lesion in the CT
    assert tensor_lab.shape[0] == 1
    if tumor_mask.sum() == 0:
        #print('Fell back to random crop')
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        crop_organ = 'random'
    else:
        # Crop around a random tumor voxel
        #which classes have tumor? sum last 3 dimensions
        positives = tumor_mask.sum(dim=(-3,-2,-1))>0
        possibilities = [i for i in range(positives.shape[0]) if positives[i]]
        #choose one of the possibilities
        chosen_class = possibilities[ torch.randint(0, len(possibilities), (1,)).item() ]
        #print('Chosen tumor class:',chosen_class)
        tumor_mask = tumor_mask[chosen_class]
        crop_organ = lesion_classes[chosen_class]
        tumor_voxels = torch.nonzero(tumor_mask)
        center = tumor_voxels[torch.randint(0, len(tumor_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    if return_crop_organ:
        return tensor_img, tensor_lab, crop_organ
    else:
        return tensor_img, tensor_lab


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 746-787 (denoise_mask)
def denoise_mask(mask_3d, iterations=2, connected_component=True):
    """
    Perform `iterations` binary erosions + `iterations` binary dilations,
    then AND with the original mask to remove small/noisy regions.
    Then keep only the largest connected component of the result.
    """
    device = mask_3d.device
    #check if mask is torch tensor
    if isinstance(mask_3d, torch.Tensor):
        np_mask = mask_3d.cpu().numpy().astype(bool)
    else:
        np_mask = mask_3d.astype(bool)

    # 1) Morphological denoise
    eroded  = binary_erosion(np_mask, iterations=iterations)
    dilated = binary_dilation(eroded,  iterations=iterations)
    final   = dilated & np_mask  # shape: (D,H,W), bool

    if connected_component:
        # 2) Label connected components in `final`
        labeled, num_components = label(final)  # labeled: int array with [1..num_components] labels

        if num_components == 0:
            # No foreground at all
            refined_mask = torch.from_numpy(final).to(device)
        elif num_components == 1:
            # Only one component, so it's already the largest
            refined_mask = torch.from_numpy(final).to(device)
        else:
            # More than one => pick largest
            # counts[i] = number of voxels with label i
            counts = np.bincount(labeled.ravel())
            # Index 0 is background, so ignore it by zeroing it out.
            counts[0] = 0  
            largest_label = np.argmax(counts)     # The label with the most voxels
            largest_mask = (labeled == largest_label)
            refined_mask = torch.from_numpy(largest_mask).to(device)
    else:
        # No connected component analysis, just return the mask
        refined_mask = torch.from_numpy(final).to(device)

    return refined_mask


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 790-1019 (crop_foreground_3d)
def crop_foreground_3d(tensor_ct, tensor_lab, foreground, crop_size, margin=1, refine_iterations=3, rand=True):
    """
    Crops a 3D CT & binary label around the label's nonzero region, returning EXACT [d,h,w].
    
    If rand=True, the bounding box is randomly shifted within the volume if possible.
    If rand=False, it is centered if possible.

    1) If label is empty => return "zero mask"
    2) If bounding box is bigger than crop_size => morphological denoise => 
       if still doesn't fit => return "mask does not fit crop size"
    3) If bounding box <= crop_size => compute the valid range of random shifts
       for each dimension. If no valid shift is possible => "mask does not fit crop size"
    4) Otherwise, pick a random shift and return (cropped_ct, cropped_label).

    Args:
        tensor_ct (torch.Tensor): shape [D,H,W] or [1,D,H,W]
        foreground (torch.Tensor): shape [D,H,W] or [1,D,H,W], binary
        
        crop_size (tuple/list): (d,h,w)
        margin (int or tuple): extra margin
        refine_iterations (int): # of erosions/dilations

    Returns:
        (cropped_ct, cropped_label) or
        "zero mask" or
        "mask does not fit crop size"
    """

    ##### 1) Unify shapes #####
    if tensor_ct.ndim == 3:
        D, H, W = tensor_ct.shape
        ct_has_channel = False
        ct_has_batch = False
    elif tensor_ct.ndim == 4 and tensor_ct.shape[0] == 1:
        _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = False
    elif tensor_ct.ndim == 5 and tensor_ct.shape[0] == 1 and tensor_ct.shape[1] == 1:
        _, _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = True
    else:
        raise ValueError(f"CT must be [D,H,W] or [1,D,H,W] or [1,1,D,H,W], got {tensor_ct.shape}")

    # --- replace the old squeeze block ----------------------------------------
    if foreground.ndim == 4 and foreground.shape[0] == 1:
        # foreground is [1, D, H, W]  → drop the leading 1
        label_3d = foreground[0].clone()
    elif foreground.ndim == 3:
        label_3d = foreground.clone()
    else:
        raise ValueError(
            f"Foreground must be [D,H,W] or [1,D,H,W], got {foreground.shape}"
        )
    
    backup_foreground = foreground.clone()
        
    # Check empty
    if torch.count_nonzero(label_3d) == 0:
        return "zero mask"

    ##### 2) Get bounding box #####
    coords = torch.nonzero(label_3d, as_tuple=False)
    zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
    ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
    xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

    if isinstance(margin, int):
        margin = (margin, margin, margin)
    mz, my, mx = margin

    # Apply margin---this is the foreground bounding box
    zmin = max(zmin - mz, 0)
    zmax = min(zmax + mz, D - 1)
    ymin = max(ymin - my, 0)
    ymax = min(ymax + my, H - 1)
    xmin = max(xmin - mx, 0)
    xmax = min(xmax + mx, W - 1)
    
    # After applying margin and clamping:
    if xmin > xmax:   xmin, xmax = xmax, xmin
    if ymin > ymax:   ymin, ymax = ymax, ymin
    if zmin > zmax:   zmin, zmax = zmax, zmin

    desired_d, desired_h, desired_w = crop_size

    def bbox_dim(z0, z1, y0, y1, x0, x1):
        return (z1 - z0 + 1), (y1 - y0 + 1), (x1 - x0 + 1)

    bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)

    # Check if bounding box is bigger
    if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
        # Attempt morphological denoise
        refined = denoise_mask(label_3d, iterations=refine_iterations)
        label_3d = refined.clone()
        if torch.count_nonzero(refined) == 0:
            return "zero mask"

        # Recompute bounding box
        coords = torch.nonzero(refined, as_tuple=False)
        zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
        ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
        xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

        zmin = max(zmin - mz, 0)
        zmax = min(zmax + mz, D - 1)
        ymin = max(ymin - my, 0)
        ymax = min(ymax + my, H - 1)
        xmin = max(xmin - mx, 0)
        xmax = min(xmax + mx, W - 1)
        
        if xmin > xmax:   xmin, xmax = xmax, xmin
        if ymin > ymax:   ymin, ymax = ymax, ymin
        if zmin > zmax:   zmin, zmax = zmax, zmin

        bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)
        if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
            return "mask does not fit crop size"

    ##### 3) We know bounding box is <= crop_size. Let's find valid shifts. #####

    # We want subvolume [zstart : zstart+desired_d-1] to fully contain [zmin : zmax].
    # => zstart <= zmin
    # => zstart+desired_d-1 >= zmax => zstart >= zmax - (desired_d-1)
    # So zstart in [ zmax-(desired_d-1), zmin ]
    # Also zstart cannot be negative, and zstart+desired_d-1 cannot extend beyound the volume.
    # We'll define a helper:

    def valid_shifts_1D(min_bb, max_bb, vol_size, crop_size):
        """
        Returns a range (low, high) of all valid starting positions 
        such that [start : start+crop_size-1] fully contains [min_bb : max_bb]
        and stays within [0, vol_size-1].
        If there's no valid integer in [low, high], no shift is possible.
        """
        min_start = max_bb - (crop_size - 1)  # bounding box forced at the 'end'
        max_start = min_bb                    # bounding box forced at the 'start'

        # clamp to [0, vol_size - crop_size]
        lower_bound = 0
        upper_bound = vol_size - crop_size

        # intersection
        final_low = max(min_start, lower_bound)
        final_high = min(max_start, upper_bound)
        return int(final_low), int(final_high)

    # z dimension
    z_low, z_high = valid_shifts_1D(zmin, zmax, D, desired_d)
    # y dimension
    y_low, y_high = valid_shifts_1D(ymin, ymax, H, desired_h)
    # x dimension
    x_low, x_high = valid_shifts_1D(xmin, xmax, W, desired_w)

    # If any dimension has final_low > final_high, 
    # there's no integer that can satisfy bounding box constraints.
    if z_low > z_high or y_low > y_high or x_low > x_high:
        return "mask does not fit crop size"

    # Helper to pick shift in one dimension
    # If there's no valid shift (low>high), we 'crop in place' by placing bounding box at zmin
    # (clamped so we stay inside [0, vol_size - crop_size]).
    def pick_shift_1d(low, high, bb_min, vol_size, csize, rand_flag):
        if low > high:
            # No shift range => just place bounding box at bb_min (clamp to valid range)
            return max(0, min(bb_min, vol_size - csize))
        else:
            if rand_flag:
                return random.randint(int(low), int(high))
            else:
                return (low + high) // 2

    ##### 4) Pick the shift (or no shift if none is possible) #####
    z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, rand)
    y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, rand)
    x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, rand)

    z_end = z_start + desired_d
    y_end = y_start + desired_h
    x_end = x_start + desired_w
    
    def dbg(dim, low, high, start, bb_min, bb_max, size):
        print(f"{dim}:  bb=({bb_min},{bb_max})  "
            f"shift_range=[{low},{high}]  chosen={start}  "
            f"crop=({start},{start+size-1})")
    #dbg('z', z_low, z_high, z_start, zmin, zmax, desired_d)
    #dbg('y', y_low, y_high, y_start, ymin, ymax, desired_h)
    #dbg('x', x_low, x_high, x_start, xmin, xmax, desired_w)

    # Now we check if indeed we are inside the volume
    if z_end > D or y_end > H or x_end > W:
        raise ValueError(f"Crop failed. Why? It should not fail here.")

    ##### 5) Final Crop #####
    if ct_has_channel and not ct_has_batch:
        cropped_ct = tensor_ct[:, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, z_start:z_end, y_start:y_end, x_start:x_end]
    elif ct_has_channel and ct_has_batch:
        cropped_ct = tensor_ct[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
    else:
        cropped_ct = tensor_ct[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[z_start:z_end, y_start:y_end, x_start:x_end]

    if cropped_ct.shape[-3:] != (desired_d, desired_h, desired_w):
        raise ValueError(f"Crop failed, got {cropped_ct.shape[-3:]}. Why? It should not fail here.")
    
    cropped_fg = label_3d[z_start:z_end, y_start:y_end, x_start:x_end]
    if torch.count_nonzero(cropped_fg) == 0:
        #is the original foreground 0?
        print('Original foreground total:', torch.count_nonzero(label_3d))
        print('Cropped foreground total:',torch.count_nonzero(cropped_fg))
        #check for inplace changes at foreground
        print('Inplace changes in foreground:',(not torch.equal(foreground,backup_foreground)))
        #is the problem in random??
        z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, False)
        y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, False)
        x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, False)

        z_end = z_start + desired_d
        y_end = y_start + desired_h
        x_end = x_start + desired_w
        
        cropped_fg_deter = label_3d[z_start:z_end, y_start:y_end, x_start:x_end]
        
        print('Deter foreground total:',torch.count_nonzero(cropped_fg_deter))
        raise ValueError("zero mask after crop")

    return (cropped_ct, cropped_label, cropped_fg)


# --- copied verbatim: R-Super@0cac36f rsuper_train/training/augmentation.py lines 1023-1075 (pad_volume_pair)
def pad_volume_pair(input_tensor: torch.Tensor, label_tensor: torch.Tensor, 
                    desired_d: int, desired_h: int, desired_w: int):
    """
    Pads both the input_tensor and label_tensor along their last three dimensions 
    (assumed to be depth, height, width) with zeros on both sides if any of the dimensions 
    is smaller than the specified desired size. The same padding is applied to both tensors.
    
    If no padding is needed (i.e. all spatial dimensions are at least the desired sizes), 
    the original tensors are returned.
    
    Args:
        input_tensor (torch.Tensor): Tensor of shape (..., D, H, W).
        label_tensor (torch.Tensor): Tensor of shape (..., D, H, W). Must have the same shape
                                     in the last three dimensions as input_tensor.
        desired_d (int): Desired depth.
        desired_h (int): Desired height.
        desired_w (int): Desired width.
    
    Returns:
        tuple: (padded_input, padded_label), both with spatial dimensions at least 
               (desired_d, desired_h, desired_w).
    """
    # Check that the spatial dimensions of the two tensors match.
    if input_tensor.shape[-3:] != label_tensor.shape[-3:]:
        raise ValueError("The input and label tensors must have the same spatial dimensions.")
    
    current_d, current_h, current_w = input_tensor.shape[-3:]
    
    # Compute total padding required for each spatial dimension.
    pad_d = max(0, desired_d - current_d)
    pad_h = max(0, desired_h - current_h)
    pad_w = max(0, desired_w - current_w)
    
    # If no padding is needed, return the original tensors.
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return input_tensor, label_tensor
    
    # Compute symmetric padding amounts.
    pad_d_left = pad_d // 2
    pad_d_right = pad_d - pad_d_left
    pad_h_left = pad_h // 2
    pad_h_right = pad_h - pad_h_left
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    # F.pad expects a tuple with padding for the last dimension first.
    # For 3 dimensions, the tuple is: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
    padding = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
    
    padded_input = F.pad(input_tensor, padding, mode="constant", value=0)
    padded_label = F.pad(label_tensor, padding, mode="constant", value=0)
    
    return padded_input, padded_label

# #############################################################################################
# PART 2 - OUR CODE (not from the repo): U-Net, data loading of mask / report cases, training
# loop and sliding-window inference. Each piece states which repo code it stands in for.
# #############################################################################################
import argparse, json, time, glob, types
import pandas as pd

CLASSES = ['pancreas', 'pancreas_body', 'pancreas_head', 'pancreas_tail', 'pancreatic_lesion']   # sorted, as in the repo
LESION = CLASSES.index('pancreatic_lesion')
SEG_IDX = {'head': CLASSES.index('pancreas_head'), 'body': CLASSES.index('pancreas_body'),
           'tail': CLASSES.index('pancreas_tail')}   # report_npz gt has the first 4 channels in the same order
TUMOR_CLASS_IDX = [LESION]                         # lesion channel(s), as self.lesion_classes in the repo loader
FOREGROUND_IDX = [CLASSES.index('pancreas')]       # organ used for organ crops (repo: forg = ['pancreas'])


# ============================================================================== model
class ConvBlock(nn.Module):
    """Two 3x3x3 conv + InstanceNorm + LeakyReLU (nnU-Net style)."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Plain 3D U-Net (stand-in for MedFormer). Output format follows MedFormer.prepare_return:
    {'segmentation': [final_logits, aux_logits]}, both at input resolution. The auxiliary (deep-supervision)
    head sits on the 1/4-resolution decoder level and is upsampled trilinearly, as MedFormer's aux_out."""
    def __init__(self, n_classes=5, ch=(32, 64, 128, 256, 320)):
        super().__init__()
        self.enc = nn.ModuleList([ConvBlock(1, ch[0])] + [ConvBlock(ch[i - 1], ch[i], stride=2) for i in range(1, len(ch))])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i], ch[i - 1], 2, stride=2) for i in range(len(ch) - 1, 0, -1)])
        self.dec = nn.ModuleList([ConvBlock(2 * ch[i - 1], ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        self.out = nn.Conv3d(ch[0], n_classes, 1)
        self.aux_out = nn.Conv3d(ch[2], n_classes, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()
        aux = None
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
            if x.shape[1] == self.aux_out.in_channels and aux is None:
                aux = self.aux_out(x)
        out = self.out(x)
        aux = F.interpolate(aux, size=out.shape[-3:], mode='trilinear', align_corners=True)
        return {'segmentation': [out, aux]}


# ============================================================================== data
def npy_cache(npz_path, cache_dir):
    """npz (compressed) -> npy once, so every sample can be read with mmap like the repo's npy path."""
    out = os.path.join(cache_dir, os.path.basename(npz_path).replace('.npz', '.npy'))
    if not os.path.exists(out):
        arr = np.load(npz_path)['arr_0']
        tmp = out + '.tmp.npy'
        np.save(tmp, arr)
        os.replace(tmp, out)
    return out


def report_segments(rows):
    """Report rows of one CT -> (groups, flat, options), mirroring get_tumor_segment_labels():
    groups  : location of each tumour as a list, e.g. ['head'] or ['head', 'body'] for 'head / body'
    flat    : every sub-segment that contains a tumour   (-> unknown voxels of the lesion channel)
    options : sub-segment(s) the crop may be centred on; segments that share a tumour are merged
              (repo: 'subseg_with_only_known_sizes'; all our synthetic tumours have a known size)."""
    groups = []
    for loc in rows['Standardized Location'].tolist():
        if isinstance(loc, str) and loc not in ('', 'u'):
            g = loc.split(' / ')
            if g not in groups:
                groups.append(g)
    flat = sorted(set(s for g in groups for s in g))
    options = []
    for s in flat:
        options.append(sorted(set(x for g in groups if s in g for x in g)))
    return groups, flat, options


def tumors_in_segment(rows, chosen):
    """estimate_tumor_volume(): tumours whose whole location lies in the chosen sub-segment(s).
    Size 'd1 x d2' -> d3 = (d1 + d2) / 2 and ellipsoid volume 4/3*pi*(d1/2)(d2/2)(d3/2) in mm^3 = voxels (1 mm grid).
    Returns volumes (10,) and diameters (10, 3), zero-padded, as the repo."""
    vols, dias = [], []
    for _, r in rows.iterrows():
        loc = r['Standardized Location']
        if not isinstance(loc, str) or loc.lower() == 'u' or loc == '':
            continue
        if all(l in chosen for l in loc.split(' / ')):
            size = str(r['Tumor Size (mm)'])
            if 'x' not in size:
                d = float(size); vols.append(4 / 3 * math.pi * (d / 2) ** 3); dias.append([d, d, d])
            else:
                s = [float(v) for v in size.split(' x ')]
                if len(s) == 2:
                    s.append(sum(s) / 2)
                vols.append(4 / 3 * math.pi * (s[0] / 2) * (s[1] / 2) * (s[2] / 2)); dias.append(s)
    vols, dias = vols[:10], dias[:10]
    vols += [0.0] * (10 - len(vols)); dias += [[0.0, 0.0, 0.0]] * (10 - len(dias))
    return torch.tensor(vols).float(), torch.tensor(dias).float()


class RSuperDataset(torch.utils.data.Dataset):
    """Stand-in for training/dataset/dim3/dataset_abdomenatlas_UFO.py (and dataset_abdomenatlas.py when
    there are no report cases). Every item is a dict with the keys calculate_loss() expects:
      image (1,d,h,w), label (C,d,h,w), unk_channels (C,..), mask = chosen_segment_mask (C,..),
      volumes (10,), diameters (10,3).
    * mask cases  : random_crop_on_tumor() (90 % on a tumour voxel, 5 % organ, 5 % background); 40 % of the
                    time a larger crop is rotated (+-30 deg, small shear) and centre-cropped, as in the repo.
    * report cases: 10 % random crop; otherwise crop_foreground_3d() around a reported sub-segment
                    (the 'chosen segment'). Lesion channel = 0 but UNKNOWN (unk) inside every reported
                    sub-segment; chosen_segment_mask = chosen sub-segment in the lesion channel;
                    volumes/diameters = report tumours inside the chosen sub-segment.
    * mask and report lists are balanced by repeating the shorter one (repo: balance_supervision).
    * intensity augmentation as the repo: each of 6 transforms with p = 0.3."""

    def __init__(self, mask_ids, mask_dir, report_ids=(), report_dir=None, reports_csv=None, crop=96,
                 cache_dir='/tmp/npy_cache', length=10 ** 7, seed=0):
        self.crop, self.length = crop, length
        os.makedirs(cache_dir, exist_ok=True)
        t0 = time.time()
        mask = [(npy_cache(f'{mask_dir}/{b}.npz', cache_dir), npy_cache(f'{mask_dir}/{b}_gt.npz', cache_dir), b, False)
                for b in mask_ids]
        rep = [(npy_cache(f'{report_dir}/{b}.npz', cache_dir), npy_cache(f'{report_dir}/{b}_gt.npz', cache_dir), b, True)
               for b in report_ids]
        print(f'npy cache ready ({len(mask)} mask + {len(rep)} report cases, {time.time()-t0:.0f}s)', flush=True)
        self.reports = {}
        if rep:
            df = pd.read_csv(reports_csv)
            for b in report_ids:
                rows = df[df['BDMAP ID'] == b]
                no_lesion = len(rows) == 0 or bool(rows['no lesion'].astype(str).str.lower().eq('true').all())
                self.reports[b] = None if no_lesion else rows
            if len(mask) < len(rep):                     # balance_supervision
                mask = mask + random.Random(seed).choices(mask, k=len(rep) - len(mask))
            elif len(rep) < len(mask):
                rep = rep + random.Random(seed).choices(rep, k=len(mask) - len(rep))
        self.items = mask + rep
        random.Random(seed).shuffle(self.items)
        self.stats = dict(on_segment=0, failed=0, random=0)

    def __len__(self):
        return self.length

    def _load(self, i):
        img_p, lab_p, bd, is_rep = self.items[i % len(self.items)]
        img = torch.from_numpy(np.array(np.load(img_p, mmap_mode='r'))).float()[None, None]
        lab = torch.from_numpy(np.array(np.load(lab_p, mmap_mode='r')).astype(np.int8))[None]
        d = self.crop
        img, lab = pad_volume_pair(img, lab, d + 20, d + 40, d + 40)
        return img, lab, bd, is_rep

    def _mask_case(self, img, lab):
        d = self.crop
        if np.random.random() < 0.4:      # random_crop_on_tumor() with rotation, as dataset_abdomenatlas*.py
            img, lab = random_crop_on_tumor(img, lab, TUMOR_CLASS_IDX, d + 20, d + 40, d + 40, True, foreground_classes=FOREGROUND_IDX)
            img, lab = random_scale_rotate_translate_3d(img, lab, [0, 0, 0], [30, 30, 30], [0, 0, 0])
            img, lab = crop_3d(img, lab, [d, d, d], mode='center')
        else:
            img, lab = random_crop_on_tumor(img, lab, TUMOR_CLASS_IDX, d, d, d, True, foreground_classes=FOREGROUND_IDX)
        lab = lab[0].float()
        z = torch.zeros_like(lab)
        return img[0], lab, z, z.clone(), torch.zeros(10), torch.zeros(10, 3)

    def _report_case(self, img, lab, bd):
        d = self.crop
        rows = self.reports[bd]
        # 4 organ channels -> 5 channels (lesion channel added as 0; assign_labels())
        lab5 = torch.zeros((1, len(CLASSES)) + tuple(lab.shape[-3:]), dtype=lab.dtype)
        lab5[:, :4] = lab[:, :4]
        chosen = 'random'
        if rows is None:                                  # 'no lesion' report -> random/organ crop, lesion known = 0
            img, lab5 = random_crop_on_tumor(img, lab5, [], d, d, d, False, foreground_classes=FOREGROUND_IDX)
            flat = []
        else:
            groups, flat, options = report_segments(rows)
            if np.random.random() < 0.1:                  # 'Random crop by chance'
                img, lab5 = crop_3d(img, lab5, [d, d, d], mode='random')
                self.stats['random'] += 1
            else:
                opts = list(options)
                while opts:
                    seg = random.choice(opts)
                    fg = sum(lab5[0, SEG_IDX[s]].float() for s in seg) > 0
                    out = crop_foreground_3d(tensor_ct=img, tensor_lab=lab5, foreground=fg, crop_size=[d, d, d])
                    if isinstance(out, tuple):
                        img, lab5, _ = out
                        chosen = seg
                        self.stats['on_segment'] += 1
                        break
                    opts = [o for o in opts if o != seg]
                if chosen == 'random':                    # segment does not fit the crop -> organ crop, no report loss
                    self.stats['failed'] += 1
                    img, lab5 = random_crop_on_tumor(img, lab5, [], d, d, d, False, foreground_classes=FOREGROUND_IDX)
        lab5 = lab5[0].float()
        unk = torch.zeros_like(lab5)
        for s in flat:                                    # every reported sub-segment in the crop: tumour location unknown
            unk[LESION] = torch.maximum(unk[LESION], (lab5[SEG_IDX[s]] > 0).float())
        mask = torch.zeros_like(lab5)
        if chosen != 'random':
            vols, dias = tumors_in_segment(rows, chosen)
            mask[LESION] = (sum(lab5[SEG_IDX[s]] for s in chosen) > 0).float()   # get_chosen_segment_mask()
        else:
            vols, dias = torch.zeros(10), torch.zeros(10, 3)
        return img[0], lab5, unk, mask, vols, dias

    def __getitem__(self, idx):
        img, lab, bd, is_rep = self._load(idx)
        if is_rep:
            img, lab, unk, mask, vols, dias = self._report_case(img, lab, bd)
        else:
            img, lab, unk, mask, vols, dias = self._mask_case(img, lab)
        x = img[None]
        for p, f in [(0.3, lambda t: brightness_multiply(t, multiply_range=[0.7, 1.3])),
                     (0.3, lambda t: brightness_additive(t, std=0.1)),
                     (0.3, lambda t: gamma(t, gamma_range=[0.7, 1.5])),
                     (0.3, lambda t: contrast(t, contrast_range=[0.7, 1.3])),
                     (0.3, lambda t: gaussian_blur(t, sigma_range=[0.5, 1.5])),
                     (0.3, lambda t: gaussian_noise(t, std=np.random.random() * 0.2))]:
            if np.random.random() < p:
                x = f(x)
        return {'image': x[0].float(), 'label': lab, 'unk_channels': unk, 'volumes': vols.float(),
                'mask': mask, 'diameters': dias.float(), 'is_report': torch.tensor(float(is_rep))}


# ============================================================================== training
def lr_multiplier(epoch, warmup, max_epoch):
    """exp_lr_scheduler_with_warmup() in rsuper_train/training/utils.py: exponential warm-up, then poly(0.9)."""
    if warmup and 0 <= epoch <= warmup:
        return 1.0 if epoch == warmup else math.exp(10.0 * (float(epoch) / float(warmup) - 1.))
    return (1.0 - epoch / max_epoch) ** 0.9


@torch.no_grad()
def update_ema(model, ema, alpha, step):
    """update_ema_variables() in rsuper_train/training/utils.py."""
    alpha = min(1 - 1 / (step + 1), alpha)
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(alpha).add_(p.detach(), alpha=1 - alpha)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def loss_args(cfg):
    """The args fields calculate_loss() reads; values = repo defaults unless set in cfg."""
    return types.SimpleNamespace(loss=cfg['loss'], aux_weight=[0.5, 0.5], seg_loss=1.0,
                                 report_volume_loss_basic=cfg['report_weight'],
                                 volume_loss_tolerance=cfg['volume_tolerance'], ball_bce_weight=1.0, ball_dice_weight=1.0,
                                 multi_ch_tumor=False, stardard_ce_ball=False, classification_branch=False,
                                 ball_volume_margin=0.2)


def train(cfg):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    out_dir = cfg['out_dir']; os.makedirs(out_dir, exist_ok=True)
    sanity = os.path.join(cfg.get('sanity_dir') or out_dir, 'sanity'); os.makedirs(sanity, exist_ok=True)
    os.chdir(sanity)       # the copied losses write their first few debug volumes to the working directory
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = lambda p: pd.read_csv(p)['BDMAP ID'].tolist() if p else []
    ds = RSuperDataset(ids(cfg['mask_ids']), cfg['mask_dir'], ids(cfg.get('report_ids')), cfg.get('report_dir'),
                       cfg.get('reports'), crop=cfg['crop'], cache_dir=cfg['cache_dir'], seed=cfg['seed'])
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg['batch'], num_workers=cfg['workers'], pin_memory=True,
                                     persistent_workers=cfg['workers'] > 0,
                                     worker_init_fn=lambda w: (np.random.seed((torch.initial_seed() + w) % 2 ** 32),
                                                               random.seed(torch.initial_seed() + w)))
    net = UNet3D(len(CLASSES)).to(dev)
    ema = UNet3D(len(CLASSES)).to(dev); ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.05)   # config: adamw, wd 0.05
    use_amp = dev.type == 'cuda' and cfg.get('amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ck = os.path.join(out_dir, 'latest.pth')
    start_epoch, step = 0, 0
    if os.path.exists(ck):                                       # resume
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s['model']); ema.load_state_dict(s['ema']); opt.load_state_dict(s['opt'])
        scaler.load_state_dict(s['scaler']); start_epoch, step = s['epoch'], s['step']
        print(f'resumed from {ck} at epoch {start_epoch}', flush=True)
    elif cfg.get('init_from'):                                   # fine-tune from the masks-only model (README: --pretrain)
        s = torch.load(cfg['init_from'], map_location=dev, weights_only=False)
        net.load_state_dict(s['model']); ema.load_state_dict(s['ema'])
        print(f'initialised from {cfg["init_from"]} (epoch {s["epoch"]})', flush=True)
    model = nn.DataParallel(net) if dev.type == 'cuda' and torch.cuda.device_count() > 1 else net
    largs = loss_args(cfg)
    print(f'device {dev} x{max(1, torch.cuda.device_count())} | amp {use_amp} | loss {largs.loss} | report weight '
          f'{largs.report_volume_loss_basic} | tolerance {largs.volume_loss_tolerance} | params '
          f'{sum(p.numel() for p in net.parameters())/1e6:.1f} M', flush=True)
    it = iter(dl)
    for epoch in range(start_epoch, cfg['epochs']):
        m = lr_multiplier(epoch, cfg['warmup'], cfg['epochs'])
        for g in opt.param_groups:
            g['lr'] = cfg['lr'] * m
        print(f'Starting epoch {epoch+1}/{cfg["epochs"]} lr {cfg["lr"]*m:.2e}', flush=True)
        net.train(); t0 = time.time(); agg = {}
        for i in range(cfg['iters']):
            b = next(it)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model(b['image'])
            out = {'segmentation': [o.float() for o in out['segmentation']]}    # losses in fp32
            L = calculate_loss(model_output=out, label=b['label'], unk_voxels=b['unk_channels'], args=largs, matcher=None,
                               chosen_segment_mask=b['mask'], tumor_volumes_report=b['volumes'],
                               tumor_diameters=b['diameters'], classes=CLASSES, input_tensor=b['image'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(L['overall']).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            scaler.step(opt); scaler.update()
            step += 1
            update_ema(net, ema, 0.99, step)
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if (i + 1) % cfg['print_freq'] == 0:
                print(f'  ep {epoch+1} it {i+1}/{cfg["iters"]} ' + ' '.join(f'{k} {v/(i+1):.4f}' for k, v in agg.items())
                      + f' | {(time.time()-t0)/(i+1):.2f} s/it', flush=True)
        torch.save(dict(model=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch + 1, step=step, cfg=cfg), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
        with open(os.path.join(out_dir, 'history.jsonl'), 'a') as f:
            f.write(json.dumps(dict(epoch=epoch + 1, sec=round(time.time() - t0, 1), **{k: v / cfg['iters'] for k, v in agg.items()})) + '\n')
        print(f'epoch {epoch+1} done in {time.time()-t0:.0f}s; ' + ' '.join(f'{k} {v/cfg["iters"]:.4f}' for k, v in agg.items())
              + f' | crops {getattr(ds, "stats", {})}', flush=True)


# ============================================================================== inference
def gaussian_importance(size, sigma_scale=0.125):
    c = (np.arange(size) - (size - 1) / 2) / (size * sigma_scale)
    g = np.exp(-0.5 * c ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return torch.from_numpy((w / w.max()).astype(np.float32)).clamp_min(1e-3)


@torch.no_grad()
def sliding_window(model, img, win, overlap=0.5, dev='cpu', amp=True):
    """Sliding-window inference with Gaussian weighting (the repo uses MONAI-style sliding windows of
    window_size with overlap). img: (D,H,W) float tensor -> sigmoid probabilities (C,D,H,W)."""
    D, H, W = img.shape
    pd_ = [max(0, win - s) for s in (D, H, W)]
    x = F.pad(img, (0, pd_[2], 0, pd_[1], 0, pd_[0]))
    Dp, Hp, Wp = x.shape
    step = max(1, int(win * (1 - overlap)))
    starts = lambda n: sorted(set(list(range(0, n - win + 1, step)) + [n - win]))
    wmap = gaussian_importance(win).to(dev)
    prob = torch.zeros((len(CLASSES), Dp, Hp, Wp), device=dev)
    norm = torch.zeros((Dp, Hp, Wp), device=dev)
    for z in starts(Dp):
        for y in starts(Hp):
            for xx in starts(Wp):
                patch = x[z:z + win, y:y + win, xx:xx + win][None, None].to(dev)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp and dev.type == 'cuda'):
                    o = model(patch)['segmentation'][0]
                prob[:, z:z + win, y:y + win, xx:xx + win] += torch.sigmoid(o.float())[0] * wmap
                norm[z:z + win, y:y + win, xx:xx + win] += wmap
    prob = prob / norm
    return prob[:, :D, :H, :W].cpu()


def predict(cfg):
    dev = torch.device(f'cuda:{cfg["gpu"]}' if torch.cuda.is_available() else 'cpu')
    s = torch.load(cfg['ckpt'], map_location=dev, weights_only=False)
    net = UNet3D(len(CLASSES)).to(dev)
    net.load_state_dict(s['ema'])                       # the repo evaluates the EMA model ('Use EMA model for evaluation')
    net.eval()
    ids = pd.read_csv(cfg['ids'])['BDMAP ID'].tolist()
    ids = ids[cfg.get('part', 0)::cfg.get('parts', 1)]
    os.makedirs(cfg['save_dir'], exist_ok=True)
    for k, bd in enumerate(ids):
        out = os.path.join(cfg['save_dir'], bd + '.npz')
        if os.path.exists(out):
            continue
        t0 = time.time()
        img = torch.from_numpy(np.load(os.path.join(cfg['img_dir'], bd + '.npz'))['arr_0']).float()
        prob = sliding_window(net, img, cfg['window'], cfg.get('overlap', 0.5), dev)
        np.savez_compressed(out, pred=(prob > 0.5).numpy().astype(np.uint8), classes=np.array(CLASSES))
        print(f'[{k+1}/{len(ids)}] {bd} {tuple(img.shape)} {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'predict'])
    ap.add_argument('--cfg', required=True, help='JSON file with the run settings')
    a = ap.parse_args()
    with open(a.cfg) as f:
        c = json.load(f)
    (train if a.mode == 'train' else predict)(c)
