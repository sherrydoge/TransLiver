import os
import numpy as np
import torch.utils.data as Data


class TDataset(Data.Dataset):
    # Dataset for test: return (img, label)
    def __init__(self, files, base_phase, root):
        # initialize
        self.files = files
        self.moving_imgs = []
        self.fix_imgs = []
        self.moving_labels = []
        self.fix_labels = []
        phases = ["artery", "delayed", "plain", "venous"]
        phases.remove(base_phase)
        for file in files:
            fpath, fname = os.path.split(file)
            for phase in phases:
                tfile = os.path.join(root, phase, fname)
                if os.path.exists(tfile):
                    fix = np.load(file)['vol']
                    mov = np.load(tfile)['vol']
                    fl = np.load(file)['seg']
                    ml = np.load(tfile)['seg']
                    self.fix_imgs.append(fix)
                    self.moving_imgs.append(mov)
                    self.fix_labels.append(fl)
                    self.moving_labels.append(ml)

    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):
        return self.moving_imgs[index].transpose(2, 0, 1)[np.newaxis, ...], self.fix_imgs[index].transpose(2, 0, 1)[np.newaxis, ...],\
               self.moving_labels[index].transpose(2, 0, 1)[np.newaxis, ...], self.fix_labels[index].transpose(2, 0, 1)[np.newaxis, ...],  # [C, D, W, H]
