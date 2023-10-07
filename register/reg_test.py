# python imports
import os
import glob

# external imports
import torch
import numpy as np
import nibabel as nib

# internal imports
from Model.config import args
from Model.model import U_Network, SpatialTransformer
from reg_postprocess import make_dir


def register_all():
    if not os.path.exists(args.reg_data_path):
        os.mkdir(args.reg_data_path)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # get shape
    vol_size = [args.slice_num, args.img_size, args.img_size]  # D, W, H

    # affine (change with dataset)
    path = args.affine_path
    data = nib.load(path)
    affine = data.affine

    # create Unet and STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_epoch.pth')))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    train_files = glob.glob(os.path.join(args.train_dir, args.base_phase, '*.npz'))
    test_files = glob.glob(os.path.join(args.test_dir, args.base_phase, '*.npz'))
    files = train_files + test_files
    phases = ["artery", "delayed", "plain", "venous"]
    phases.remove(args.base_phase)

    for file in files:
        fpath, fname = os.path.split(file)
        root, p = os.path.split(fpath)
        for phase in phases:
            des_img_path = os.path.join(args.reg_data_path, phase)
            des_label_path = os.path.join(args.reg_data_path, phase+"_label")
            if not os.path.exists(des_img_path):
                make_dir(des_img_path)
            if not os.path.exists(des_label_path):
                make_dir(des_label_path)
            tfile = os.path.join(root, phase, fname)
            if os.path.exists(tfile):
                print(fname)
                # [B, C, D, W, H]
                fix = np.load(file)['vol'].transpose(2, 0, 1)[np.newaxis, np.newaxis, ...]
                mov = np.load(tfile)['vol'].transpose(2, 0, 1)[np.newaxis, np.newaxis, ...]
                ml = np.load(tfile)['seg'].transpose(2, 0, 1)[np.newaxis, np.newaxis, ...]
                moving_img = torch.from_numpy(mov).to(device).float()
                fix_img = torch.from_numpy(fix).to(device).float()
                moving_label = torch.from_numpy(ml).to(device).float()

                pred_flow = UNet(moving_img, fix_img)
                pred_img = STN_img(moving_img, pred_flow)
                pred_label = STN_label(moving_label, pred_flow)

                m2f_name = os.path.join(des_img_path, fname.split('.')[0]+".nii.gz")
                m2fl_name = os.path.join(des_label_path, fname.split('.')[0]+".nii.gz")
                nib.save(nib.Nifti1Image(pred_img[0, 0, ...].cpu().detach().numpy().transpose(1, 2, 0), affine), m2f_name)
                nib.save(nib.Nifti1Image(pred_label[0, 0, ...].cpu().detach().numpy().transpose(1, 2, 0), affine), m2fl_name)


if __name__ == "__main__":
    register_all()
