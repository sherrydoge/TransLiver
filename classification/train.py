import os
import numpy as np
import warnings
import logging

import torch
import torch.distributed as dist
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from utils import count_parameters, make_dir, same_seed, get_cosine_schedule_with_warmup
from config import args
from datagenerator import LesionSliceDataset
from model import generate_model_mbt


def train():
    print("train begin")

    # log
    print("Preparation...")
    same_seed(args.seed)
    real_epochs = args.epochs * args.data_iter
    log_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code
    print("log_name: ", log_name)
    f = os.path.join(args.log_path, log_name + ".txt")
    logging.basicConfig(filename=f, level=logging.INFO, filemode='w',
                        format='[%(asctime)s.%(msecs)03d] %(message)s')
    logging.info(str(args))

    logging.info("the program will use %d card(s)" % args.cuda_num)
    gpus = [0]
    is_multi = False
    local_rank = 0
    if args.cuda_num > 1:
        is_multi = True
        logging.info("multi-process communication address %s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
        world_size = torch.cuda.device_count()
        if not world_size == args.cuda_num:
            logging.error("real card number does not match config: real %d config %d" % (world_size, args.cuda_num))
            return
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        gpus = [local_rank]
    else:
        torch.cuda.set_device(0)

    if local_rank == 0:
        runs_log_name = os.path.join(args.log_path, "runs")
        writer = SummaryWriter(os.path.join(runs_log_name, log_name))

    # get shape
    vol_size = (args.slice_num, args.img_size, args.img_size)  # D, H, W

    # generate data
    print("Generating data...")
    train_path = args.train_path + '_' + args.data_type
    val_path = args.val_path + '_' + args.data_type
    class_path = os.path.join(args.lesion_path, "lesion_slice_classes_organized.npy")
    train_ds = LesionSliceDataset(train_path, args.base_phase, vol_size, args.num_classes, class_path,
                                  args.transform, False, False, False, args.data_iter)
    val_ds = LesionSliceDataset(val_path, args.base_phase, vol_size, args.num_classes, class_path,
                                False, False, True, False, 1)
    if is_multi:
        train_sampler = DistributedSampler(train_ds)
        train_loader = Data.DataLoader(dataset=train_ds, batch_size=args.batch_size,
                                       sampler=train_sampler, shuffle=False, num_workers=args.num_workers)
    else:
        train_sampler = None
        train_loader = Data.DataLoader(dataset=train_ds, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers)
    val_loader = Data.DataLoader(dataset=val_ds, batch_size=1, shuffle=True)

    # set model
    print("Setting model...")
    model_paras = args.model_name.split('_')
    pretrain_path = None
    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_path, args.pretrain_name)
    model = generate_model_mbt(model_type=model_paras[0], model_scale=model_paras[1], phase_num=int(model_paras[2]),
                               bottleneck_n=int(model_paras[3]), backbone=model_paras[4], gpu_id=gpus,
                               pretrain_path=pretrain_path, nb_class=args.num_classes, drop_out=args.drop_out,
                               is_multi=is_multi, in_channel=args.in_channel)

    if local_rank == 0:
        logging.info("The model need %d parameters" % count_parameters(model))

    # set optimizer and losses
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.epochs*0.1), args.epochs)
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0
    best_acc_patient = 0
    model_save_path = os.path.join(os.path.join(args.checkpoint_path, log_name))
    if local_rank == 0:
        make_dir(model_save_path)

    # train
    print("Training...")
    for i in range(args.epochs):
        model.train()
        train_loss_record = []
        train_correct_num = 0
        if is_multi:
            train_sampler.set_epoch(i)
        for j, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
            lesions, label = data
            lesions = lesions.cuda().float()
            label = label.cuda()

            # forward
            pred = model(lesions)
            train_loss = loss_function(pred, label)
            train_loss_record.append(train_loss.item())
            pred_class = torch.max(pred, dim=1)[1]
            train_correct_num += torch.eq(pred_class, label).sum().item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        scheduler.step()

        # gather train data from all gpus
        train_loss_record = np.array(train_loss_record)
        train_loss_record = torch.from_numpy(train_loss_record).cuda()
        train_correct_num = torch.tensor(train_correct_num).cuda()
        all_loss_list = [torch.zeros_like(train_loss_record) for _ in range(args.cuda_num)]
        all_num_list = [torch.zeros_like(train_correct_num) for _ in range(args.cuda_num)]
        if is_multi:
            dist.gather(train_loss_record, all_loss_list if local_rank == 0 else None, dst=0)
            dist.gather(train_correct_num, all_num_list if local_rank == 0 else None, dst=0)
        else:
            all_loss_list[0] = train_loss_record
            all_num_list[0] = train_correct_num

        model.eval()
        val_correct_num = 0
        patient_class_label = {}
        patient_class_pred = {}
        for j, data in tqdm(enumerate(val_loader), total=len(val_loader), leave=True):
            lesions, label, patient = data
            patient = patient[0]

            # for slice data
            if patient not in patient_class_pred:
                patient_class_pred[patient] = [0] * args.num_classes
                patient_class_label[patient] = label.item()
            lesions = lesions.cuda().float()
            label = label.cuda()

            with torch.no_grad():
                pred = model(lesions)
                pred_class = torch.max(pred, dim=1)[1]
                val_correct_num += torch.eq(pred_class, label).sum().item()
                # for slice data
                patient_class_pred[patient][pred_class.item()] += 1

        if not local_rank == 0:
            continue  # only log and save model in process 0

        # evaluate model performance
        sum_train_loss = 0.0
        cnt_train_loss = 0
        for tl in all_loss_list:
            sum_train_loss += sum(tl.cpu().detach().numpy())
            cnt_train_loss += len(tl.cpu().detach().numpy())
        mean_train_loss = sum_train_loss / cnt_train_loss
        sum_correct_num = 0
        for cn in all_num_list:
            sum_correct_num += cn.item()
        train_acc = sum_correct_num / len(train_ds)  # slice accuracy for train
        val_acc = val_correct_num / len(val_ds)  # slice accuracy for val
        # for slice data
        val_correct_num_patient = [0] * args.num_classes
        for p in patient_class_pred:
            max_num = max(patient_class_pred[p])
            res = []
            for index, num in enumerate(patient_class_pred[p]):
                if num == max_num:
                    res.append(index)
            if patient_class_label[p] in res:
                val_correct_num_patient[patient_class_label[p]] += 1
        val_acc_patient = sum(val_correct_num_patient) / len(patient_class_label)  # lesion accuracy for val

        # log
        writer.add_scalar('train_loss', mean_train_loss, i)
        writer.add_scalar('train_acc', train_acc, i)
        logging.info("local rank:%d  epoch:%d  train_loss:%f  train_acc:%f" %
                     (local_rank, i, mean_train_loss, train_acc))
        writer.add_scalar('val_acc', val_acc, i)
        logging.info("local rank:%d  epoch:%d  val_acc:%f  val_acc_patient:%f" %
                     (local_rank, i, val_acc, val_acc_patient))

        # save the model with the best slice accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = i
            save_file_name = os.path.join(model_save_path, "best_epoch.pth")
            if is_multi:
                torch.save(model.module.state_dict(), save_file_name)  # save model.module
            else:
                torch.save(model.state_dict(), save_file_name)
            logging.info("best model in epoch %d with acc %f is saved." % (best_epoch, best_acc))

        # save the model with the best lesion accuracy
        if val_acc_patient > best_acc_patient:
            best_acc_patient = val_acc_patient
            best_epoch_patient = i
            save_file_name = os.path.join(model_save_path, "best_epoch_patient.pth")
            if is_multi:
                torch.save(model.module.state_dict(), save_file_name)  # save model.module
            else:
                torch.save(model.state_dict(), save_file_name)
            logging.info("best model in epoch %d with acc %f is saved." % (best_epoch_patient, best_acc_patient))

        # save the last model
        if i == args.epochs-1:
            save_file_name = os.path.join(model_save_path, "last_epoch.pth")
            if is_multi:
                torch.save(model.module.state_dict(), save_file_name)  # save model.module
            else:
                torch.save(model.state_dict(), save_file_name)
            logging.info("last model in epoch %d with acc %f is saved." % (i, val_acc))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        train()
