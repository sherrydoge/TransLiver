import os
import shutil
import random
from glob import glob
import nibabel as nib
import numpy as np

from scipy.ndimage.interpolation import zoom

from Model.config import args


def make_dir(dir_name):
    # make directory recursively
    dpath, dname = os.path.split(dir_name)
    if not os.path.exists(dpath):
        make_dir(dpath)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def MinMax(image):
    image = image - np.mean(image)  # zero average
    if np.max(image) - np.min(image) != 0:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize
    return image


def nii_preprocess():
    root = args.reg_root
    phases = ["artery", "delayed", "plain", "venous"]
    for phase in phases:
        print("Processing  " +phase)
        imgs = glob(os.path.join(root, phase, "*.nii.gz"))
        labels = glob(os.path.join(root, phase + "_label", "*.nii.gz"))
        crop(imgs, 0, 512, 0, 512, 32)
        crop(labels, 0, 512, 0, 512, 32)


def crop(imgs, x1, x2, y1, y2, z1):
    # crop, clip, normalize
    print(str(len(imgs)) +" imgs totally")
    for index, img in enumerate(imgs):
        if index % 100 == 0:
            print(str(index) +" imgs done!")
        data = nib.load(img)
        affine = data.affine
        x, y, z = data.shape
        mat = data.get_fdata()
        if z >= z1:
            mat = mat[x1:x2, y1:y2, -z1:]
        else:
            mat = mat[x1:x2, y1:y2, -z:]
        x, y, z = mat.shape

        target_size = [224, 224, 32]  # downsample to 224*224*32
        if "label" in img:
            mat = zoom(mat, (target_size[0] / x, target_size[1] / y, target_size[2] / z), order=0)
        else:
            mat = zoom(mat, (target_size[0] / x, target_size[1] / y, target_size[2] / z), order=3)
            mat = np.clip(mat, -140, 200)
            mat = MinMax(mat)
        new_data = nib.Nifti1Image(mat, affine)
        nib.save(new_data, img)


def nii2npz():
    root = args.reg_root
    img_dirs = ["artery", "delayed", "plain", "venous"]
    label_dirs = ["artery_label", "delayed_label", "plain_label", "venous_label"]
    target_root = os.path.join(root, "npz")
    for index, phase in enumerate(img_dirs):
        print("Processing "+phase)
        make_dir(os.path.join(target_root, phase))
        img_path = os.path.join(root, phase)
        label_path = os.path.join(root, label_dirs[index])
        patients = os.listdir(img_path)
        for patient in patients:
            pid = patient.split('.')[0]
            target_path = os.path.join(target_root, phase, pid+".npz")
            img = nib.load(os.path.join(img_path, patient))
            label = nib.load(os.path.join(label_path, patient))
            idata = img.get_fdata()
            ldata = label.get_fdata()
            np.savez(target_path, vol=idata, seg=ldata)


def divide():
    # divide data into train and test
    root = os.path.join(args.reg_root, "npz")
    img_dirs = ["artery", "delayed", "plain", "venous"]
    base_phase = "artery"
    img_path = os.path.join(root, base_phase)
    patients = os.listdir(img_path)
    random.shuffle(patients)
    train_ratio = 0.8
    offset = int(len(patients)*train_ratio)
    train_patients = patients[:offset]
    test_patients = patients[offset:]
    for patient in train_patients:
        print("Processing train data")
        for phase in img_dirs:
            make_dir(os.path.join(root, "train", phase))
            source_path = os.path.join(root, phase, patient)
            target_path = os.path.join(root, "train", phase, patient)
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
    for patient in test_patients:
        print("Processing test data")
        for phase in img_dirs:
            make_dir(os.path.join(root, "test", phase))
            source_path = os.path.join(root, phase, patient)
            target_path = os.path.join(root, "test", phase, patient)
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
