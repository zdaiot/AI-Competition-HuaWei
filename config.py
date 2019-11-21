import json
import argparse
from argparse import Namespace


def get_classify_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=tuple, default=[512, 512], help='image size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=80, help='epoch')

    parser.add_argument('--augmentation_flag', type=bool, default=False,
                        help='if true, use augmentation method in train set')
    parser.add_argument('--n_splits', type=int, default=1, help='n_splits_fold')
    parser.add_argument('--val_size', type=float, default=0.2, help='the ratio of val data when n_splits=1.')
    # model set 
    parser.add_argument('--model_type', type=str, default='resnet50', help='resnet50')
    
    # model hyper-parameters
    parser.add_argument('--num_classes', type=int, default=54)
    parser.add_argument('--lr', type=float, default=4e-5, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay in optimizer')
    # 学习率衰减策略
    parser.add_argument('--lr_scheduler', type=str, default='CosineLR', help='lr scheduler')
    parser.add_argument('--lr_step_size', type=str, default=30, help='step_size for StepLR scheduler')
    parser.add_argument('--restart_step', type=str, default=80, help='T_max for CosineAnnealingLR scheduler')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    # 损失函数
    parser.add_argument('--loss_name', type=str, default='1.0*CrossEntropy',
                        help='Select the loss function, CrossEntropy/SmoothCrossEntropy')

    # 路径
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset_root', type=str, default='data/huawei_data/train_data')

    config = parser.parse_args()

    return config
