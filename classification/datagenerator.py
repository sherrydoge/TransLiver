import os

import numpy as np
import torch
import torch.utils.data as Data
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate


def lesion_transform(lesions, rotate_prob=0.1, crop_prob=0.1, shift_prob=0.1, scale_prob=0.1):
    # data augmentation
    spatial_dim = len(lesions[0].shape)
    img_size = [224, 224]

    # flip
    flip_choice = np.random.randint(0, spatial_dim+1)
    for i in range(len(lesions)):
        if flip_choice < spatial_dim:
            lesions[i] = np.flip(lesions[i], axis=flip_choice)
        else:
            lesions[i] = lesions[i]

    # rotate 90
    k = np.random.randint(0, 4)
    for i in range(len(lesions)):
        lesions[i] = np.rot90(lesions[i], k)

    # rotate
    rotate_flag = np.random.uniform()
    if rotate_flag < rotate_prob:
        angle_choice = np.random.uniform(-1, 1)
        for i in range(len(lesions)):
            lesions[i] = rotate(lesions[i], 45*angle_choice, order=0, reshape=False, mode="reflect")

    # crop
    crop_flag = np.random.uniform()
    if crop_flag < crop_prob:
        crop_scale = 0.75
        crop_size = [int(x*crop_scale) for x in img_size]
        start_point = []
        for j in range(spatial_dim):
            start_point.append(np.random.randint(0, img_size[j] - crop_size[j]))
        for i in range(len(lesions)):
            lesions[i] = lesions[i][start_point[0]:start_point[0]+crop_size[0],
                                    start_point[1]:start_point[1]+crop_size[1]]
            lesions[i] = zoom(lesions[i], (img_size[0]/crop_size[0], img_size[1]/crop_size[1]))

    # shift
    shift_flag = np.random.uniform()
    if shift_flag < shift_prob:
        offset = np.random.uniform(-0.5, 0.5)
        for i in range(len(lesions)):
            lesions[i] = lesions[i] + offset

    # scale
    scale_flag = np.random.uniform()
    if scale_flag < scale_prob:
        factor = np.random.uniform(0.5, 1.5)
        for i in range(len(lesions)):
            lesions[i] = lesions[i] * factor


def zscore(lesions):
    # z-score normalization
    for i in range(len(lesions)):
        lesion_mean = np.mean(lesions[i])
        lesion_std = np.std(lesions[i])
        lesions[i] = (lesions[i]-lesion_mean)/lesion_std


class LesionSliceDataset(Data.Dataset):

    # Dataset for 4 phase lesions slice
    def __init__(self, lesion_path: str, base_phase: str, img_shape: tuple, num_classes: int,
                 class_path: str, transform: bool, no_phase_data: bool, is_test: bool, slice_position: bool,
                 data_iter: int):
        super().__init__()
        self.lesions_list = []
        self.class_list = []
        self.patient_list = []
        self.base_phase = base_phase
        self.num_classes = num_classes
        self.transform = transform
        self.is_test = is_test  # return lesion id when test
        lesion_class_dict = np.load(class_path, allow_pickle=True).item()
        phases = ["artery", "delayed", "plain", "venous"]
        for i, phase in enumerate(phases):
            if phase == base_phase:
                phases[0], phases[i] = phases[i], phases[0]
                break

        lesions = os.listdir(os.path.join(lesion_path, base_phase))
        for lesion in lesions:
            patient_lesions = []
            lesion_slice = 0
            for phase in phases:
                source_path = os.path.join(lesion_path, phase, lesion)
                lesion_img = nib.load(source_path).get_fdata()
                lesion_slice = lesion_img.shape[2]
                patient_lesions.append(lesion_img)
            for i in range(lesion_slice):
                patient_lesions_slice = []
                for lesion_img in patient_lesions:
                    patient_lesions_slice.append(lesion_img[..., i])
                zscore(patient_lesions_slice)
                if no_phase_data:
                    for single_lesion in patient_lesions_slice:
                        for _ in range(data_iter):
                            self.lesions_list.append([single_lesion])
                            self.class_list.append(lesion_class_dict[lesion.split('.')[0]])
                            self.patient_list.append(lesion.split('.')[0])
                else:
                    for _ in range(data_iter):
                        self.lesions_list.append(patient_lesions_slice)
                        self.class_list.append(lesion_class_dict[lesion.split('.')[0]])
                        self.patient_list.append(lesion.split('.')[0])

    def __getitem__(self, index):
        lesions = self.lesions_list[index].copy()
        cla = int(self.class_list[index])-1
        patient = self.patient_list[index]

        if self.transform:
            lesion_transform(lesions)

        for i in range(len(lesions)):
            lesions[i] = lesions[i].transpose(1, 0)  # [H, W]
            lesions[i] = torch.Tensor(lesions[i].copy()).unsqueeze(0).unsqueeze(0).float()  # [P, C, H, W]

        if self.is_test:
            return torch.cat(lesions, 0), cla, patient

        return torch.cat(lesions, 0), cla

    def __len__(self):
        return len(self.class_list)
