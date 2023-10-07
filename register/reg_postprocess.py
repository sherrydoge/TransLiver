import os
from glob import glob
import nibabel as nib
import numpy as np

from Model.config import args


def make_dir(dir_name):
    # make directory recursively
    dpath, dname = os.path.split(dir_name)
    if not os.path.exists(dpath):
        make_dir(dpath)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def update_limits(limits, val, i, j, k):
    if i < limits[val][0]:
        limits[val][0] = i
    if i > limits[val][1]:
        limits[val][1] = i
    if j < limits[val][2]:
        limits[val][2] = j
    if j > limits[val][3]:
        limits[val][3] = j
    if k < limits[val][4]:
        limits[val][4] = k
    if k > limits[val][5]:
        limits[val][5] = k


def lesion_divide(label):
    # divide the lesions in one label and generate the mask(or the region)
    masks = {}  # generate masks for each lesion
    limits = {}  # record the limits of each lesion
    x, y, z = label.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                lab_val = int(label[i, j, k])
                if lab_val > 0:
                    if lab_val in masks:
                        masks[lab_val][i, j, k] = 1
                        update_limits(limits, lab_val, i, j, k)
                    else:
                        masks[lab_val] = np.zeros(label.shape, dtype=int)
                        limits[lab_val] = [x, 0, y, 0, z, 0]
                        masks[lab_val][i, j, k] = 1
                        update_limits(limits, lab_val, i, j, k)
    return masks, limits


def lesion_extract_phases():
    # extract lesions for multi-phases
    source_root = args.reg_data_path
    target_root = args.lesion_path
    phases = ["artery", "delayed", "plain", "venous"]
    lesion_limits = {}
    for phase in phases:
        print("Processing " + phase)
        img_path = os.path.join(source_root, phase)
        label_path = os.path.join(source_root, phase+"_label")
        target_path = os.path.join(target_root, phase)
        make_dir(target_path)
        patients = os.listdir(img_path)
        for patient in patients:
            pid = patient.split('.')[0]
            img_data = nib.load(os.path.join(img_path, patient))
            affine = img_data.affine
            img = img_data.get_fdata()
            label_data = nib.load(os.path.join(label_path, patient))
            label = label_data.get_fdata()
            masks, limits = lesion_divide(label)
            for mask_id in masks:
                mask = masks[mask_id]
                if mask.shape != img.shape:
                    continue
                limit = limits[mask_id]
                lesion_limits[phase+"_"+pid+"_"+str(mask_id)] = limit
                mask_window = np.zeros(img.shape, dtype=int)  # generate window mask
                for i in range(limit[0], limit[1]+1):
                    for j in range(limit[2], limit[3]+1):
                        for k in range(limit[4], limit[5]+1):
                            mask_window[i][j][k] = 1
                lesion = img*mask_window
                if phase+"_"+pid+"_"+str(mask_id) not in lesion_limits:
                    print(phase+"_"+pid+"_"+str(mask_id))
                nib.save(nib.Nifti1Image(lesion, affine), os.path.join(target_path, pid+"_"+str(mask_id)+".nii.gz"))
    np.save(os.path.join(target_root, "lesion_limits.npy"), lesion_limits)  # type: ignore


def lesion_window_dice(la, lb):
    mask_shape = (224, 224, 32)
    maska = np.zeros(mask_shape, dtype=int)
    maskb = np.zeros(mask_shape, dtype=int)
    for i in range(la[0], la[1]+1):
        for j in range(la[2], la[3]+1):
            for k in range(la[4], la[5]+1):
                maska[i][j][k] = 1
    for i in range(lb[0], lb[1]+1):
        for j in range(lb[2], lb[3]+1):
            for k in range(lb[4], lb[5]+1):
                maskb[i][j][k] = 1
    smooth = 1e-5
    maska = maska.flatten()
    maskb = maskb.flatten()
    intersection = (maska*maskb).sum()
    return (2. * intersection + smooth) / (maska.sum() + maskb.sum() + smooth)


def lesion_organize():
    # renumber the lesions in different phases to make sure the same lesions have the same value
    # organize lesions into 4-phase
    root = args.lesion_path
    phases = ["artery", "delayed", "plain", "venous"]
    remain_phases = ["artery", "delayed", "plain", "venous"]
    min_matched = {}  # the minimum distance of lesions in remain phases
    same_lesion_dict = {}  # record origin id of same lesions
    same_lesions = []  # ["phase_id", "phase_id", "phase_id", "phase_id"]
    threshold = 0.3  # only when lesion dice is higher than threshold, we consider two lesions the same
    lesion_limits = np.load(os.path.join(root, "lesion_limits.npy"), allow_pickle=True).item()

    for phase in phases:
        source_path = os.path.join(root, phase)
        lesions = os.listdir(source_path)
        remain_phases.remove(phase)
        for lesion in lesions:
            if phase+"_"+lesion in min_matched:
                continue
            id = lesion.split('_')[0]
            same_lesion = [phase + "_" + lesion]
            limit = lesion_limits[phase+'_'+lesion.split('.')[0]]  # only compute distance in this window
            same_lesion_cnt = len(same_lesions)
            for remain_phase in remain_phases:
                obj_lesions = glob(os.path.join(root, remain_phase, id+"*.nii.gz"))
                dices = []
                for obj_lesion in obj_lesions:
                    fpath, fname = os.path.split(obj_lesion)
                    obj_limit = lesion_limits[remain_phase+"_"+fname.split('.')[0]]
                    dices.append(lesion_window_dice(limit, obj_limit))
                if len(dices) == 0:
                    continue
                opt_dis = max(dices)
                opt_i = dices.index(opt_dis)
                fpath, fname = os.path.split(obj_lesions[opt_i])
                if opt_dis > threshold:
                    if remain_phase+"_"+fname in min_matched:
                        index, dis = min_matched[remain_phase+"_"+fname]
                        if opt_dis > dis:
                            # if the object lesion has a better choice, choose it
                            min_matched[remain_phase+"_"+fname] = (same_lesion_cnt, opt_dis)
                            same_lesions[index].remove(remain_phase+"_"+fname)
                            same_lesion.append(remain_phase+"_"+fname)
                    else:
                        min_matched[remain_phase+"_"+fname] = (same_lesion_cnt, opt_dis)
                        same_lesion.append(remain_phase+"_"+fname)
            same_lesions.append(same_lesion)
        break  # only for 4 phases

    for same_lesion in same_lesions:
        if len(same_lesion) < 4:
            continue
        patient = same_lesion[0].split('_')[1]
        lesion_id = same_lesion[0].split('_')[2].split('.')[0]
        same_lesion_dict[patient+"_"+lesion_id] = []

    np.save(os.path.join(root, "same_lesions_dice"+str(int(threshold*100))+".npy"), same_lesion_dict)  # type: ignore
