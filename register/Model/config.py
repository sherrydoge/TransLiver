import argparse

parser = argparse.ArgumentParser()

# public parameters
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='3')
parser.add_argument("--img-size", type=int, help="image size",
                    dest="img_size", default=224)
parser.add_argument("--slice-num", type=int, help="slice number",
                    dest="slice_num", default=32)
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result-dir", type=str, help="results folder",
                    dest="result_dir", default='./Result')
parser.add_argument("--base-phase", type=str, help="results folder",
                    dest="base_phase", default='artery')
parser.add_argument("--seed", type=int, help="random seed",
                    dest="seed", default=0)
parser.add_argument("--affine-path", type=str, help="affine file path",
                    dest="affine_path", default="/path/to/affine")
parser.add_argument("--reg-root", type=str, help="unregistered file path",
                    dest="reg_root", default="/path/to/dataset")
parser.add_argument("--lesion-path", type=str, help="lesion data path",
                    dest="lesion_path", default="/path/to/lesions")

# train parameters
parser.add_argument("--train-dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="/path/to/train")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=0.5)
parser.add_argument("--dice-weight", type=float, help="dice loss weight",
                    dest="dice_weight", default=0.05)
parser.add_argument("--batch-size", type=int, help="batch_size",
                    dest="batch_size", default=2)
parser.add_argument("--model-dir", type=str, help="models folder",
                    dest="model_dir", default='./Checkpoint')
parser.add_argument("--log-dir", type=str, help="logs folder",
                    dest="log_dir", default='./Log')
parser.add_argument("--epochs", type=int, default=150, dest="epochs", help="max epochs")
parser.add_argument("--val-ratio", type=float, default=0.2, dest="val_ratio", help="validate data ratio")

# test parameters
parser.add_argument("--test-dir", type=str, help="test data directory",
                    dest="test_dir", default='/path/to/test')
parser.add_argument("--checkpoint-path", type=str, help="model weight file",
                    dest="checkpoint_path", default="./Checkpoint/best_epoch.pth")
parser.add_argument("--reg-data-path", type=str, help="data path after registration",
                    dest="reg_data_path", default="/path/to/registered/data")

args = parser.parse_args()
