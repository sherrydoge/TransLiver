import os
import numpy as np
import warnings
import logging

import torch
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc

from utils import same_seed
from config import args
from datagenerator import LesionSliceDataset
from model import generate_model_mbt


def test():
    print("test begin")
    same_seed(args.seed)

    # log
    real_epochs = args.epochs * args.data_iter
    pth_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code
    log_name = args.model_name + "_e" + str(real_epochs) + "_bs" + str(args.batch_size) + "_lr" + str(args.lr) + "_" + args.data_type + "_" + args.phase_code + "_test"
    print("log_name: ", log_name)
    f = os.path.join(args.log_path, log_name + ".txt")
    logging.basicConfig(filename=f, level=logging.INFO, filemode='a',
                        format='[%(asctime)s.%(msecs)03d] %(message)s')

    torch.cuda.set_device(0)

    # get shape
    vol_size = (args.slice_num, args.img_size, args.img_size)  # D, H, W

    # generate data
    print("Generating data...")
    test_path = args.test_path + '_' + args.data_type
    class_path = os.path.join(args.lesion_path, "lesion_slice_classes_organized.npy")
    ds = LesionSliceDataset(test_path, args.base_phase, vol_size, args.num_classes, class_path, transform=False,
                            is_test=True, no_phase_data=False, slice_position=True, data_iter=1)
    dl = Data.DataLoader(dataset=ds, batch_size=1, shuffle=True)

    # set model
    print("Setting model...")
    model_paras = args.model_name.split('_')
    model = generate_model_mbt(model_type=model_paras[0], model_scale=model_paras[1], phase_num=int(model_paras[2]),
                               bottleneck_n=int(model_paras[3]), backbone=model_paras[4], gpu_id=[0],
                               pretrain_path=None, nb_class=args.num_classes, is_multi=False,
                               in_channel=args.in_channel)

    # choose the best model to load
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "best_epoch.pth")))
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "best_epoch_patient.pth")))
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, pth_name, "last_epoch.pth")))

    print("Testing...")
    test_correct_num = 0
    test_num_class = [0]*args.num_classes
    test_correct_num_class = [0]*args.num_classes
    patient_class_label = {}
    patient_class_pred = {}
    patient_pred_tensors = {}
    label_ls = []
    pred_ls = []
    model.eval()
    for data in dl:
        lesions, label, patient = data
        patient = patient[0]

        # for slice data
        if patient not in patient_class_pred:
            patient_class_pred[patient] = [0]*args.num_classes
            patient_class_label[patient] = label.item()
            patient_pred_tensors[patient] = np.zeros((1, 7), dtype=np.float)

        lesions = lesions.cuda().float()
        label = label.cuda()
        pred = model(lesions)
        pred_class = torch.max(pred, dim=1)[1]
        pred_np = pred.cpu().detach().numpy().copy()
        label_np = np.zeros_like(pred_np)
        label_np[0][label.item()] = 1
        pred_ls.append(pred_np)
        label_ls.append(label_np)
        patient_pred_tensors[patient] += pred_np

        test_correct_num += torch.eq(pred_class, label).sum().item()
        if pred_class.item() == label.item():
            test_correct_num_class[label.item()] += 1
        test_num_class[label.item()] += 1
        # for slice data
        patient_class_pred[patient][pred_class.item()] += 1

    test_acc = test_correct_num / len(ds)
    # for slice data
    test_correct_num_patient = [0] * args.num_classes
    test_num_patient_class = [0] * args.num_classes
    test_pred_num_class = [0] * args.num_classes
    patient_label_ls = []
    patient_pred_ls = []

    # calculate the accuracy of each class
    for p in patient_class_pred:
        max_num = max(patient_class_pred[p])
        res = []
        for index, num in enumerate(patient_class_pred[p]):
            if num == max_num:
                res.append(index)
        if patient_class_label[p] in res:
            test_correct_num_patient[patient_class_label[p]] += 1
            test_pred_num_class[patient_class_label[p]] += 1
        else:
            test_pred_num_class[res[0]] += 1
        test_num_patient_class[patient_class_label[p]] += 1
        patient_pred_np = patient_pred_tensors[p] / sum(patient_class_pred[p])
        patient_label_np = np.zeros_like(patient_pred_np)
        patient_label_np[0][patient_class_label[p]] = 1
        patient_pred_ls.append(patient_pred_np)
        patient_label_ls.append(patient_label_np)

    test_acc_patient = sum(test_correct_num_patient) / len(patient_class_label)

    test_acc_class = [0.0]*args.num_classes
    test_acc_patient_class = [0.0]*args.num_classes
    test_pre_patient_class = [0.0]*args.num_classes
    test_spe_patient_class = [0.0]*args.num_classes
    test_f1_patient_class = [0.0]*args.num_classes
    patient_labels = np.concatenate(patient_label_ls, axis=0)
    patient_scores = np.concatenate(patient_pred_ls, axis=0)
    patient_fpr = [0.0] * args.num_classes
    patient_tpr = [0.0] * args.num_classes
    patient_roc_auc = [0.0] * args.num_classes
    for i in range(args.num_classes):
        if not test_num_class[i] == 0:
            test_acc_class[i] = test_correct_num_class[i] / test_num_class[i]
            test_acc_patient_class[i] = test_correct_num_patient[i] / test_num_patient_class[i]
            test_pre_patient_class[i] = test_correct_num_patient[i] / test_pred_num_class[i] if test_pred_num_class[i] != 0 else 0
            test_spe_patient_class[i] = 1 - (test_pred_num_class[i]-test_correct_num_patient[i]) / (sum(test_num_patient_class)-test_num_patient_class[i])
            test_f1_patient_class[i] = 2*test_acc_patient_class[i]*test_pre_patient_class[i]/(test_acc_patient_class[i]+test_pre_patient_class[i]) if test_acc_patient_class[i]+test_pre_patient_class[i] != 0 else 0
            patient_fpr[i], patient_tpr[i], _ = roc_curve(patient_labels[:, i], patient_scores[:, i])
            patient_roc_auc[i] = auc(patient_fpr[i], patient_tpr[i])

    logging.info("test accuracy: %f" % test_acc)
    logging.info("test class accuracy: " + str(test_acc_class))
    logging.info("test patient accuracy: %f" % test_acc_patient)
    logging.info("test patient class accuracy: " + str(test_acc_patient_class))
    logging.info("test patient mean precision: " + str(sum(test_pre_patient_class)/len(test_pre_patient_class)))
    logging.info("test patient class precision: " + str(test_pre_patient_class))
    logging.info("test patient mean sensitivity: " + str(sum(test_acc_patient_class) / len(test_acc_patient_class)))
    logging.info("test patient mean specificity: " + str(sum(test_spe_patient_class) / len(test_spe_patient_class)))
    logging.info("test patient class specificity: " + str(test_spe_patient_class))
    logging.info("test patient mean f1: " + str(sum(test_f1_patient_class) / len(test_f1_patient_class)))
    logging.info("test patient class f1: " + str(test_f1_patient_class))
    logging.info("test patient mean auc: " + str(sum(patient_roc_auc) / len(patient_roc_auc)))
    logging.info("test patient class auc: " + str(patient_roc_auc))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        test()
