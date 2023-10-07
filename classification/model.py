import logging

import timm
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from models.vit_timm import create_mbt


def generate_model_mbt(model_type="mbt", model_scale="base", model_size=224, patch_size=16,
                       no_cuda=False, gpu_id=[0], is_multi=False, pretrain_path=None, nb_class=7, drop_out=0.0,
                       phase_num=4, bottleneck_n=4, backbone="vit", in_channel=1):
    assert model_type in [
        'mbt'
    ]
    assert model_scale in [
        'base'
    ]
    assert model_size in [224]
    assert patch_size in [16]

    model_name = model_type + '_' + model_scale + '_phase' + str(phase_num) + '_bottleneck' + str(bottleneck_n) + \
                     '_' + backbone

    # model = timm.create_model(model_name, pretrained=False if pretrain_path is None else True,
    #                           pretrain_path=pretrain_path, num_classes=nb_class,
    #                           in_chans=in_channel, drop_rate=drop_out)
    model = create_mbt(model_name, pretrained=False if pretrain_path is None else True,
                       pretrain_path=pretrain_path, num_classes=nb_class,
                       in_chans=in_channel, drop_rate=drop_out)

    if not no_cuda:
        model = model.cuda()
        if is_multi:
            model = SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=gpu_id, output_device=gpu_id[0])
            if gpu_id[0] == 0:
                logging.info("Converted model to use Synchronized BatchNorm.")

    return model
