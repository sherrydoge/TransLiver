# python imports
import os
import glob
import warnings
import math

# external imports
import torch
import logging
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import nibabel as nib

# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import TDataset
from Model.model import U_Network, SpatialTransformer


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # log
    log_name = args.base_phase + "_" + str(args.epochs) + "_" + str(args.lr) + "_" + str(args.alpha) + "_" + str(args.dice_weight)
    print("log_name: ", log_name)
    f = os.path.join(args.log_dir, log_name + ".txt")
    logging.basicConfig(filename=f, level=logging.INFO, filemode='w',
                        format='[%(asctime)s.%(msecs)03d] %(message)s')
    logging.info(str(args))

    runs_log_name = os.path.join(args.log_dir, "runs")
    writer = SummaryWriter(runs_log_name)

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
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    print("UNet: ", count_parameters(UNet))
    print("STN_img: ", count_parameters(STN_img))
    print("STN_label: ", count_parameters(STN_label))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.mse_loss
    dice_loss_fn = losses.BinaryDiceLoss()
    grad_loss_fn = losses.gradient_loss
    best_loss = math.inf
    best_epoch = 0
    best_fix, best_moving, best_m2f = None, None, None

    # Get all the names of the training data
    fix_files = glob.glob(os.path.join(args.train_dir, args.base_phase, '*.npz'))
    DS = TDataset(files=fix_files, base_phase=args.base_phase, root=args.train_dir)
    print("Number of training image pairs: ", len(DS))
    val_len = int(len(DS)*args.val_ratio)
    train_len = len(DS)-val_len
    trainDS, valDS = Data.random_split(dataset=DS, lengths=[train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    trainDL = Data.DataLoader(trainDS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valDL = Data.DataLoader(valDS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    moving_img, fix_img, m2f = None, None, None

    # Training loop
    for i in range(args.epochs):
        UNet.train()
        STN_img.train()
        STN_label.train()
        loss_record = []
        sim_loss_record = []
        dice_loss_record = []
        grad_loss_record = []
        for j, data in tqdm(enumerate(trainDL), total=len(trainDL), leave=True):
            moving_img, fix_img, moving_label, fix_label = data
            moving_img = moving_img.to(device).float()
            fix_img = fix_img.to(device).float()
            moving_label = moving_label.to(device).float()
            fix_label = fix_label.to(device).float()

            flow_m2f = UNet(moving_img, fix_img)
            m2f = STN_img(moving_img, flow_m2f)
            m2fl = STN_label(moving_label, flow_m2f)

            # Calculate loss
            sim_loss = sim_loss_fn(m2f, fix_img)
            dice_loss = dice_loss_fn(m2fl, fix_label)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.dice_weight * dice_loss + args.alpha * grad_loss
            sim_loss_record.append(sim_loss.item())
            dice_loss_record.append(dice_loss.item())
            grad_loss_record.append(grad_loss.item())
            loss_record.append(loss.item())

            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        mean_loss = sum(loss_record)/len(loss_record)
        mean_sim_loss = sum(sim_loss_record)/len(sim_loss_record)
        mean_dice_loss = sum(dice_loss_record)/len(dice_loss_record)
        mean_grad_loss = sum(grad_loss_record)/len(grad_loss_record)
        writer.add_scalar('loss', mean_loss, i)
        writer.add_scalar('sim_loss', mean_sim_loss, i)
        writer.add_scalar('dice_loss', mean_dice_loss, i)
        writer.add_scalar('grad_loss', mean_grad_loss, i)
        logging.info("epoch:%d  loss:%f  sim:%f  dice:%f  grad:%f" % (i, mean_loss, mean_sim_loss, mean_dice_loss, mean_grad_loss))

        UNet.eval()
        STN_img.eval()
        STN_label.eval()
        val_loss_record = []
        for data in valDL:
            moving_img, fix_img, moving_label, fix_label = data
            moving_img = moving_img.to(device).float()
            fix_img = fix_img.to(device).float()
            moving_label = moving_label.to(device).float()
            fix_label = fix_label.to(device).float()
            with torch.no_grad():
                flow_m2f = UNet(moving_img, fix_img)
                m2f = STN_img(moving_img, flow_m2f)
                m2fl = STN_label(moving_label, flow_m2f)
                val_loss = sim_loss_fn(m2f, fix_img) + args.dice_weight * dice_loss_fn(m2fl, fix_label)
                val_loss_record.append(val_loss)
        mean_val_loss = sum(val_loss_record)/len(val_loss_record)
        logging.info("epoch:%d  val:%f" % (i, mean_val_loss))
        writer.add_scalar('val_loss', mean_val_loss, i)

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_epoch = i
            best_fix = fix_img
            best_moving = moving_img
            best_m2f = m2f
            save_file_name = os.path.join(args.model_dir, 'best_epoch.pth')
            torch.save(UNet.state_dict(), save_file_name)
            print("best model have saved")

    if best_fix is not None and best_moving is not None and best_m2f is not None:
        # Save images
        f_name = "best_epoch" + "_f.nii.gz"
        m_name = "best_epoch" + "_m.nii.gz"
        m2f_name = "best_epoch" + "_m2f.nii.gz"
        nib.save(nib.Nifti1Image(best_fix[0, 0, ...].cpu().detach().numpy().transpose(1, 2, 0), affine), os.path.join(args.result_dir, f_name))
        nib.save(nib.Nifti1Image(best_moving[0, 0, ...].cpu().detach().numpy().transpose(1, 2, 0), affine), os.path.join(args.result_dir, m_name))
        nib.save(nib.Nifti1Image(best_m2f[0, 0, ...].cpu().detach().numpy().transpose(1, 2, 0), affine), os.path.join(args.result_dir, m2f_name))
        logging.info("epoch%d is the best" % (best_epoch))
        print("warped images have saved.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
