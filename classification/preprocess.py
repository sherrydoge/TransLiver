import shutil
import random
import os

import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

from config import args
from utils import make_dir


def update_lesion_limit(limit_organized, limit):
    for i in range(len(limit)):
        if limit[i] < limit_organized[i]:
            limit_organized[i] = limit[i]


def lesion_limit_square(limit):
    # turn limit slice region to square
    size = max(limit[1]-limit[0], limit[3]-limit[2])
    if not limit[1]-limit[0] == size:
        pad1 = int((size-(limit[1]-limit[0]))/2)
        pad2 = size-(limit[1]-limit[0])-pad1
        limit[0] -= pad1
        limit[0] = max(limit[0], 0)
        limit[1] += pad2
        limit[1] = min(limit[1], 223)
    if not limit[3]-limit[2] == size:
        pad1 = int((size-(limit[3]-limit[2]))/2)
        pad2 = size-(limit[3]-limit[2])-pad1
        limit[2] -= pad1
        limit[2] = max(limit[2], 0)
        limit[3] += pad2
        limit[3] = min(limit[3], 223)


def lesion_limit_padding(limit):
    if limit[0] >= args.padding_length:
        limit[0] -= args.padding_length
    if limit[1] < args.img_size-args.padding_length:
        limit[1] += args.padding_length
    if limit[2] >= args.padding_length:
        limit[2] -= args.padding_length
    if limit[3] < args.img_size-args.padding_length:
        limit[3] += args.padding_length


def generate_lesions():
    # generate lesions input according to config info
    root = args.lesion_path
    lesion_classes = np.load(os.path.join(root, "lesion_classes.npy"), allow_pickle=True).item()
    lesion_limits = np.load(os.path.join(root, "lesion_limits.npy"), allow_pickle=True).item()
    same_lesions = np.load(os.path.join(root, "same_lesions_dice"+str(args.lesion_dice_threshold)+".npy"), allow_pickle=True).item()
    lesion_classes_organized = {}
    lesion_limits_organized = {}

    for base_lesion in same_lesions:
        flag = False  # if the flag is True, the lesions are not the same
        same_lesion = same_lesions[base_lesion]
        lesion_class = None
        lesion_limit = [args.img_size, 0, args.img_size, 0, 0, args.slice_num]  # for slice data
        for lesion in same_lesion:
            limit = lesion_limits[lesion.split('.')[0]]
            update_lesion_limit(lesion_limit, limit)
            phase = lesion.split('_')[0]
            patient = lesion.split('_')[1]
            lesion_id = lesion.split('_')[2].split('.')[0]
            if lesion_class is None:
                lesion_class = lesion_classes[phase][patient][lesion_id]["class"]
            else:
                # same lesion should belong to the same class
                if not lesion_class == lesion_classes[phase][patient][lesion_id]["class"]:
                    flag = True
                    break
        if flag:
            continue
        lesion_limit_square(lesion_limit)
        lesion_limit_padding(lesion_limit)
        if lesion_limit[5]-lesion_limit[4] < 0 or lesion_limit[3]-lesion_limit[2] < 0 or lesion_limit[1]-lesion_limit[0] < 0:
            continue
        lesion_classes_organized[base_lesion] = lesion_class
        lesion_limits_organized[base_lesion] = lesion_limit

    np.save(os.path.join(root, "lesion_slice_classes_organized.npy"), lesion_classes_organized)  # type: ignore
    np.save(os.path.join(root, "lesion_slice_limits_organized.npy"), lesion_limits_organized)  # type: ignore


def divide_lesions():
    # divide lesions into train dataset and test dataset
    root = args.lesion_path
    reg_root = args.reg_path
    lesion_limits = np.load(os.path.join(root, "lesion_slice_limits_organized.npy"), allow_pickle=True).item()
    same_lesions = np.load(os.path.join(root, "same_lesions_dice" + str(args.lesion_dice_threshold) + ".npy"),
                           allow_pickle=True).item()

    lesions = list(lesion_limits.keys())
    class_dict_path = os.path.join(root, "lesion_slice_classes_organized.npy")
    class_dict = np.load(class_dict_path, allow_pickle=True).item()
    class_lesions = []
    for i in range(7):
        class_lesions.append([])
    for lesion in lesions:
        class_lesions[int(class_dict[lesion.split('.')[0]])-1].append(lesion)
    train_lesions = []
    test_lesions = []
    for i in range(7):
        random.shuffle(class_lesions[i])
        offset = int(len(class_lesions[i]) * 0.8)
        train_lesions += class_lesions[i][:offset]
        test_lesions += class_lesions[i][offset:]
    print(len(train_lesions))
    print(len(test_lesions))

    phases = ["artery", "delayed", "plain", "venous"]
    for phase in phases:
        make_dir(os.path.join(args.train_path, phase))
        make_dir(os.path.join(args.test_path, phase))

    for base_lesion in train_lesions:
        same_lesion = same_lesions[base_lesion]
        limit = lesion_limits[base_lesion]
        for lesion in same_lesion:
            phase = lesion.split('_')[0]
            patient_name = lesion.split('_')[1]
            source_path = os.path.join(reg_root, phase, patient_name+".nii.gz")
            target_path = os.path.join(args.train_path, phase, base_lesion+".nii.gz")
            source_data = nib.load(source_path)
            affine = source_data.affine
            source_img = source_data.get_fdata()
            target_img = source_img[limit[0]:limit[1]+1, limit[2]:limit[3]+1, limit[4]:limit[5]+1]
            nib.save(nib.Nifti1Image(target_img, affine), target_path)

    for base_lesion in test_lesions:
        same_lesion = same_lesions[base_lesion]
        limit = lesion_limits[base_lesion]
        for lesion in same_lesion:
            phase = lesion.split('_')[0]
            patient_name = lesion.split('_')[1]
            source_path = os.path.join(reg_root, phase, patient_name+".nii.gz")
            target_path = os.path.join(args.test_path, phase, base_lesion+".nii.gz")
            source_data = nib.load(source_path)
            affine = source_data.affine
            source_img = source_data.get_fdata()
            target_img = source_img[limit[0]:limit[1]+1, limit[2]:limit[3]+1, limit[4]:limit[5]+1]
            nib.save(nib.Nifti1Image(target_img, affine), target_path)


def train_val_split():
    phases = ["artery", "delayed", "plain", "venous"]
    class_dict_path = os.path.join(args.lesion_path, "lesion_slice_classes_organized.npy")
    class_dict = np.load(class_dict_path, allow_pickle=True).item()
    if os.path.exists(args.val_path):
        tp = os.path.join(args.train_path, args.base_phase)
        vp = os.path.join(args.val_path, args.base_phase)
        train_lesions = os.listdir(tp)
        val_lesions = os.listdir(vp)
        lesions = train_lesions + val_lesions
    else:
        tp = os.path.join(args.train_path, args.base_phase)
        train_lesions = os.listdir(tp)
        lesions = train_lesions
        for phase in phases:
            make_dir(os.path.join(args.val_path, phase))
    class_lesions = []
    for i in range(7):
        class_lesions.append([])
    for lesion in lesions:
        class_lesions[int(class_dict[lesion.split('.')[0]]) - 1].append(lesion)
    train_lesions = []
    val_lesions = []
    for i in range(7):
        random.shuffle(class_lesions[i])
        offset = int(len(class_lesions[i]) * (1-args.val_ratio))
        train_lesions += class_lesions[i][:offset]
        val_lesions += class_lesions[i][offset:]
    for lesion in train_lesions:
        for phase in phases:
            source = os.path.join(args.val_path, phase, lesion)
            target = os.path.join(args.train_path, phase, lesion)
            if not os.path.exists(target):
                shutil.move(source, target)
    for lesion in val_lesions:
        for phase in phases:
            source = os.path.join(args.train_path, phase, lesion)
            target = os.path.join(args.val_path, phase, lesion)
            if not os.path.exists(target):
                shutil.move(source, target)


def lesion_resize(lesions, target_size):
    # resize the lesion to target size
    for i in range(len(lesions)):
        if lesions[i].shape == target_size:
            continue
        lesions[i] = zoom(lesions[i], (target_size[0] / lesions[i].shape[0], target_size[1] / lesions[i].shape[1],
                                       target_size[2] / lesions[i].shape[2]))


def minmax_revert(lesions):
    # revert the lesion to original value (normalized when registered
    ct_min = -140
    ct_max = 200
    for i in range(len(lesions)):
        lesions[i] = lesions[i]*(ct_max-ct_min)+ct_min
        lesions[i] = lesions[i].astype('int32')
        lesions[i] = np.clip(lesions[i], -140, 200)


def lesion_preprocess():
    train_lesion_path = args.train_path
    val_lesion_path = args.val_path
    test_lesion_path = args.test_path
    lesion_paths = [train_lesion_path, val_lesion_path, test_lesion_path]
    base_phase = args.base_phase
    img_shape = [224, 224, 32]
    affine = np.array([[-1, 0, 0, -0], [0, -1, 0, -0], [0, 0, 1, 0], [0, 0, 0, 1]])  # may need to change
    phases = ["artery", "delayed", "plain", "venous"]
    for i, phase in enumerate(phases):
        if phase == base_phase:
            phases[0], phases[i] = phases[i], phases[0]
            break
    for lesion_path in lesion_paths:
        lesions = os.listdir(os.path.join(lesion_path, base_phase))
        for i, lesion in enumerate(lesions):
            print("%d/%d" % (i+1, len(lesions)))
            patient_lesions = []
            for phase in phases:
                source_path = os.path.join(lesion_path, phase, lesion)
                lesion_img = nib.load(source_path).get_fdata()
                patient_lesions.append(lesion_img)
            img_shape[2] = patient_lesions[0].shape[2]  # for slice data
            minmax_revert(patient_lesions)
            lesion_resize(patient_lesions, img_shape)
            for j, phase in enumerate(phases):
                nib.save(nib.Nifti1Image(patient_lesions[j], affine), os.path.join(lesion_path, phase, lesion))
