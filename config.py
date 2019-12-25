import argparse
import json
import pprint


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_classify_config():
    parser = argparse.ArgumentParser()

    # -----------------------------------------超参数设置-----------------------------------------
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
    
    # -----------------------------------------数据增强设置-----------------------------------------
    parser.add_argument('--image_size', type=json.loads, default=[320, 320],
                        help='image size, for example --image_size [256, 256], '
                             'Note that this size is also used for validation sets')
    # 多尺度设置
    parser.add_argument('--multi_scale', type=str2bool, nargs='?', const=True, default=True,
                        help='use multi scale training or not.')
    parser.add_argument('--multi_scale_size', type=json.loads,
                        default=[[224, 224], [256, 256], [288, 288], [320, 320]],
                        help='multi scale choice. For example --multi_scale_size [[224,224],[444,444]]')
    parser.add_argument('--multi_scale_interval', type=int, default=10, help='make a scale choice every [] iterations.')
    # 数据增强设置
    parser.add_argument('--augmentation_flag', type=str2bool, nargs='?', const=True, default=True,
                        help='if true, use augmentation method in train set')
    parser.add_argument('--erase_prob', type=float, default=0.0,
                        help='probability of random erase when augmentation_flag is True')
    parser.add_argument('--gray_prob', type=float, default=0.3,
                        help='probability of gray when augmentation_flag is True')
    # cut_mix设置
    parser.add_argument('--cut_mix', type=str2bool, nargs='?', const=True, default=True,
                        help='use cut mix or not.')
    parser.add_argument('--beta', type=float, default=1.0, help='beta of cut mix.')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probof cut mix.')

    # -----------------------------------------数据集设置-----------------------------------------
    parser.add_argument('--choose_dataset', type=str, choices=['only_self', 'only_official', 'combine'],
                        default='combine', help='choose dataset')
    parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
    parser.add_argument('--selected_fold', type=json.loads, default=[0], help='which folds for training')
    parser.add_argument('--val_size', type=float, default=0.2, help='the ratio of val data when n_splits=1.')
    parser.add_argument('--load_split_from_file', type=str, default='data/huawei_data/dataset_split_delete.json', 
                        help='Loading dataset split from this file')
    parser.add_argument('--dataset_from_folder', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, then load datasets distinguished by train and valid')

    # -----------------------------------------模型设置-----------------------------------------
    parser.add_argument('--model_type', type=str, default='se_resnext101_32x4d',
                        help='densenet201/efficientnet-b5/se_resnext101_32x4d')
    parser.add_argument('--drop_rate', type=float, default=0, help='dropout rate in classify module')
    parser.add_argument('--restore', type=str, default='',
                        help='Load the weight file before training.'
                             'if it is equal to `last`, load the `model_best.pth` in the last modification folder. '
                             'Otherwise, load the `model_best.pth` under the `restore` path.')
    parser.add_argument('--num_classes', type=int, default=54)

    # -----------------------------------------学习率衰减策略与优化器设置-----------------------------------------
    # 学习率衰减策略
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLR',
                        help='lr scheduler, StepLR/CosineLR/ReduceLR/MultiStepLR/CyclicLR/Flat_CosAnneal')
    parser.add_argument('--lr_step_size', type=int, default=20, help='step_size for StepLR scheduler')
    parser.add_argument('--restart_step', type=int, default=80, help='T_max for CosineAnnealingLR scheduler')
    parser.add_argument('--multi_step', type=int, nargs='+', default=[20, 35, 45], help='Milestone of MultiStepLR')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    
    # -----------------------------------------损失函数设置-----------------------------------------
    parser.add_argument('--loss_name', type=str, default='1.0*SmoothCrossEntropy',
                        help='loss name, CrossEntropy/SmoothCrossEntropy/FocalLoss/CB_Sigmoid/CB_Focal/CB_Softmax/CB_Smooth_Softmax')
    parser.add_argument('--beta_CB', type=float, default=0.9999, help='Hyperparameter for Class balanced loss.')
    parser.add_argument('--gamma', type=float, default=2, help='Hyperparameter for Focal loss.')

    # -----------------------------------------路径设置-----------------------------------------
    parser.add_argument('--train_url', type=str, default='./checkpoints',
                        help='the path to save training outputs. For example: s3://ai-competition-zdaiot/logs/')
    parser.add_argument('--data_url', type=str, default='data/huawei_data/combine')
    parser.add_argument('--model_snapshots_name', type=str, default='model_snapshots')
    parser.add_argument('--init_method', type=str)

    config = parser.parse_args()
    config.bucket_name = '/'.join(config.train_url.split('/')[:-2])

    pprint.pprint(config)
    return config


if __name__ == '__main__':
    config = get_classify_config()
    print(config.augmentation_flag)
    print(config.image_size)
    print(config.dataset_from_folder, type(config.multi_scale_size))
