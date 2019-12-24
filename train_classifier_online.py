import torch
import tqdm
import datetime
import os
import pickle
import time
import numpy as np
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import json
import codecs
try:
    import moxing as mox
except:
    print('not use moxing')

from config import get_classify_config
from solver import Solver
from utils.set_seed import seed_torch
from models.build_model import PrepareModel
from datasets.create_dataset import GetDataloader
from losses.get_loss import Loss
from utils.classification_metric import ClassificationMetric
from datasets.data_augmentation import DataAugmentation
from utils.cutmix import generate_mixed_sample
from datasets.create_dataset import multi_scale_transforms


def prepare_data_on_modelarts(args):
    """ see https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/moxing_api_doc/MoXing_API_File.md
    如果数据集存储在OBS，则需要将OBS上的数据拷贝到 ModelArts 中
    Args:
        args: 配置参数
    
    args.bucket下面有这几个文件夹：data（存放数据，包含label_id_name.json以及其他数据文件夹），project用于存放工程代码
    """
    # 将数据从OBS args.data_url拷贝到args.data_local
    args.local_data_root = '/cache/'  # a directory used for transfer data between local path and OBS path
    args.data_local = os.path.join(args.local_data_root, 'combine')
    if not os.path.exists(args.data_local):
        mox.file.copy_parallel(args.data_url, args.data_local)
    else:
        print('args.data_local: %s is already exist, skip copy' % args.data_local)

    bucket_name = args.bucket_name
    # 复制其它必要文件
    mox.file.copy(os.path.join(bucket_name, 'data', 'label_id_name.json'), args.local_data_root+'label_id_name.json')
    mox.file.copy_parallel(os.path.join(bucket_name, 'project', 'font'), os.path.join(args.local_data_root, 'font'))
    mox.file.copy_parallel(os.path.join(bucket_name, 'project', 'online-service/model/'),
                                os.path.join(args.local_data_root, 'online-service/model/'))
    if args.load_split_from_file:
        mox.file.copy(os.path.join(bucket_name, 'data', args.load_split_from_file.split('/')[-1]), 
                            args.local_data_root+args.load_split_from_file.split('/')[-1])
        args.load_split_from_file = args.local_data_root+args.load_split_from_file.split('/')[-1]
    
    # 复制扩展包
    mox.file.copy(os.path.join(bucket_name, 'data', 'torchtools-0.2.4-py3-none-any.whl'), 
                        args.local_data_root+'torchtools-0.2.4-py3-none-any.whl')
    pip = os.popen('pip install /cache/torchtools-0.2.4-py3-none-any.whl')
    print(pip.read())

    # train_local: 用于训练过程中保存的输出位置，而train_url用于移动到OBS的位置
    args.train_local = os.path.join(args.local_data_root, 'model_snapshots')
    if not os.path.exists(args.train_local):
        os.mkdir(args.train_local)

    args.tmp = os.path.join(args.local_data_root, 'tmp')
    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    return args


class TrainVal:
    def __init__(self, config, fold, train_labels_number):
        """
        Args:
            config: 配置参数
            fold: int, 当前为第几折
            train_labels_number: list, 某一折的[number_class0, number__class1, ...]
        """
        self.config = config
        self.fold = fold
        self.epoch = config.epoch
        self.num_classes = config.num_classes
        self.lr_scheduler = config.lr_scheduler
        self.cut_mix = config.cut_mix
        self.beta = config.beta
        self.cutmix_prob = config.cutmix_prob
        self.train_url = config.train_url
        self.bucket_name = config.bucket_name

        self.image_size = config.image_size
        self.multi_scale = config.multi_scale
        self.multi_scale_size = config.multi_scale_size
        self.multi_scale_interval = config.multi_scale_interval
        if self.cut_mix:
            print('Using cut mix.')
        if self.multi_scale:
            print('Using multi scale training.')
        print('USE LOSS: {}'.format(config.loss_name))

        # 拷贝预训练权重
        print("=> using pre-trained model '{}'".format(config.model_type))
        if not mox.file.exists('/home/work/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth'):
            mox.file.copy(os.path.join(self.bucket_name, 'model_zoo/se_resnext101_32x4d-3b2fe3d8.pth'),
                          '/home/work/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth')
            print('copy pre-trained model from OBS to: %s success' %
                  (os.path.abspath('/home/work/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth')))
        else:
            print('use exist pre-trained model at: %s' %
                  (os.path.abspath('/home/work/.cache/torch/checkpoints/se_resnext101_32x4d-3b2fe3d8.pth')))

        # 加载模型
        prepare_model = PrepareModel()
        self.model = prepare_model.create_model(
            model_type=config.model_type,
            classes_num=self.num_classes,
            drop_rate=config.drop_rate,
            pretrained=True
        )
        self.model = torch.nn.DataParallel(self.model).cuda()

        # 加载优化器
        self.optimizer = prepare_model.create_optimizer(config.model_type, self.model, config)

        # 加载衰减策略
        self.exp_lr_scheduler = prepare_model.create_lr_scheduler(
            self.lr_scheduler,
            self.optimizer,
            step_size=config.lr_step_size,
            restart_step=config.restart_step,
            multi_step=config.multi_step
        )

        # 加载损失函数
        self.criterion = Loss(config.model_type, config.loss_name, self.num_classes, train_labels_number, config.beta_CB, config.gamma)

        # 实例化实现各种子函数的 solver 类
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.solver = Solver(self.model, self.device)
        if config.restore:
            weight_path = os.path.join('checkpoints', config.model_type)
            if config.restore == 'last':
                lists = os.listdir(weight_path)  # 获得文件夹内所有文件
                lists.sort(key=lambda fn: os.path.getmtime(weight_path + '/' + fn))  # 按照最近修改时间排序
                weight_path = os.path.join(weight_path, lists[-1], 'model_best.pth')
            else:
                weight_path = os.path.join(weight_path, config.restore, 'model_best.pth')
            self.solver.load_checkpoint(weight_path)

        # log初始化
        self.writer, self.time_stamp = self.init_log()
        self.model_path = os.path.join(self.config.train_local, self.config.model_type, self.time_stamp)

        # 初始化分类度量准则类
        with open(config.local_data_root+'label_id_name.json', 'r', encoding='utf-8') as json_file:
            self.class_names = list(json.load(json_file).values())
        self.classification_metric = ClassificationMetric(self.class_names, self.model_path)

        self.max_accuracy_valid = 0

    def train(self, train_loader, valid_loader):
        """ 完成模型的训练，保存模型与日志
        Args:
            train_loader: 训练数据的DataLoader
            valid_loader: 验证数据的Dataloader
        """
        global_step = 0
        for epoch in range(self.epoch):
            self.model.train()
            epoch += 1
            images_number, epoch_corrects = 0, 0

            tbar = tqdm.tqdm(train_loader)
            image_size = self.image_size
            for i, (_, images, labels) in enumerate(tbar):
                if self.multi_scale:
                    if i % self.multi_scale_interval == 0:
                        image_size = random.choice(self.multi_scale_size)
                    images = multi_scale_transforms(image_size, images)
                if self.cut_mix:
                    # 使用cut_mix
                    r = np.random.rand(1)
                    if self.beta > 0 and r < self.cutmix_prob:
                        images, labels_a, labels_b, lam = generate_mixed_sample(self.beta, images, labels)
                        labels_predict = self.solver.forward(images)
                        loss = self.solver.cal_loss_cutmix(labels_predict, labels_a, labels_b, lam, self.criterion)
                    else:
                        # 网络的前向传播
                        labels_predict = self.solver.forward(images)
                        loss = self.solver.cal_loss(labels_predict, labels, self.criterion)
                else:
                    # 网络的前向传播
                    labels_predict = self.solver.forward(images)
                    loss = self.solver.cal_loss(labels_predict, labels, self.criterion)
                self.solver.backword(self.optimizer, loss)

                images_number += images.size(0)
                epoch_corrects += self.model.module.get_classify_result(labels_predict, labels, self.device).sum()
                train_acc_iteration = self.model.module.get_classify_result(labels_predict, labels, self.device).mean()

                # 保存到tensorboard，每一步存储一个
                descript = self.criterion.record_loss_iteration(self.writer.add_scalar, global_step + i)
                self.writer.add_scalar('TrainAccIteration', train_acc_iteration, global_step + i)

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'pg_%d' % group_ind + ': %.8f, ' % param_group['lr']

                descript = '[Train Fold {}][epoch: {}/{}][image_size: {}][Lr :{}][Acc: {:.4f}]'.format(
                    self.fold,
                    epoch,
                    self.epoch,
                    image_size,
                    params_groups_lr,
                    train_acc_iteration
                ) + descript

                # 对于 CyclicLR，要每一步均执行依次学习率衰减
                if self.lr_scheduler == 'CyclicLR':
                    self.exp_lr_scheduler.step()
                    self.writer.add_scalar('Lr', self.optimizer.param_groups[1]['lr'], global_step + i)

                tbar.set_description(desc=descript)

            # 写到tensorboard中
            epoch_acc = epoch_corrects / images_number
            self.writer.add_scalar('TrainAccEpoch', epoch_acc, epoch)
            if self.lr_scheduler != 'CyclicLR':
                self.writer.add_scalar('Lr', self.optimizer.param_groups[1]['lr'], epoch)
            descript = self.criterion.record_loss_epoch(len(train_loader), self.writer.add_scalar, epoch)

            # Print the log info
            print('[Finish epoch: {}/{}][Average Acc: {:.4}]'.format(epoch, self.epoch, epoch_acc) + descript)

            # 验证模型
            val_accuracy, val_loss, is_best = self.validation(valid_loader)

            # 保存参数
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_score': self.max_accuracy_valid
            }
            self.solver.save_checkpoint_online(
                os.path.join(
                    self.model_path,
                    '%s_fold%d.pth' % (self.config.model_type, self.fold)
                ),
                state,
                is_best,
                self.bucket_name
            )

            # 写到tensorboard中
            self.writer.add_scalar('ValidLoss', val_loss, epoch)
            self.writer.add_scalar('ValidAccuracy', val_accuracy, epoch)

            # 每一个epoch完毕之后，执行学习率衰减
            if self.lr_scheduler == 'ReduceLR':
                self.exp_lr_scheduler.step(val_accuracy)
            elif self.lr_scheduler != 'CyclicLR':
                self.exp_lr_scheduler.step()
            global_step += len(train_loader)
        print('BEST ACC:{}'.format(self.max_accuracy_valid))

    def validation(self, valid_loader):
        tbar = tqdm.tqdm(valid_loader)
        self.model.eval()
        labels_predict_all, labels_all = np.empty(shape=(0,)), np.empty(shape=(0,))
        epoch_loss = 0
        with torch.no_grad():
            for i, (_, images, labels) in enumerate(tbar):
                # 网络的前向传播
                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels_predict, labels, self.criterion)

                epoch_loss += loss

                # 先经过softmax函数，再经过argmax函数
                labels_predict = F.softmax(labels_predict, dim=1)
                labels_predict = torch.argmax(labels_predict, dim=1).detach().cpu().numpy()

                labels_predict_all = np.concatenate((labels_predict_all, labels_predict))
                labels_all = np.concatenate((labels_all, labels))

                descript = '[Valid][Loss: {:.4f}]'.format(loss)
                tbar.set_description(desc=descript)

            classify_report, my_confusion_matrix, acc_for_each_class, oa, average_accuracy, kappa = \
                self.classification_metric.get_metric(
                    labels_all,
                    labels_predict_all
                )

            if oa > self.max_accuracy_valid:
                is_best = True
                self.max_accuracy_valid = oa
                self.classification_metric.draw_cm_and_save_result(
                    classify_report,
                    my_confusion_matrix,
                    acc_for_each_class,
                    oa,
                    average_accuracy,
                    kappa,
                    font_fname="../font/simhei.ttf"
                )
            else:
                is_best = False

            print('OA:{}, AA:{}, Kappa:{}'.format(oa, average_accuracy, kappa))

            return oa, epoch_loss / len(tbar), is_best

    def init_log(self):
        # 保存配置信息和初始化tensorboard
        TIMESTAMP = "log-{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
        log_dir = os.path.join(self.config.train_local, self.config.model_type, TIMESTAMP)
        writer = SummaryWriter(log_dir=log_dir)
        with codecs.open(os.path.join(log_dir, 'param.json'), 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

        seed = int(time.time())
        seed_torch(seed)
        with open(os.path.join(log_dir, 'seed.pkl'), 'wb') as f:
            pickle.dump({'seed': seed}, f, -1)

        return writer, TIMESTAMP


if __name__ == "__main__":
    config = get_classify_config()
    data_root = config.data_url
    folds_split = config.n_splits
    test_size = config.val_size
    multi_scale = config.multi_scale
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    config = prepare_data_on_modelarts(config)

    if config.augmentation_flag:
        transforms = DataAugmentation(config.erase_prob, full_aug=True, gray_prob=config.gray_prob)
    else:
        transforms = None

    get_dataloader = GetDataloader(
        config.data_local,
        folds_split=folds_split,
        test_size=test_size,
        label_names_path=config.local_data_root+'label_id_name.json',
        choose_dataset=config.choose_dataset,
        load_split_from_file=config.load_split_from_file
    )

    train_dataloaders, val_dataloaders, train_labels_number_folds, _ = get_dataloader.get_dataloader(
        config.batch_size,
        config.image_size,
        mean, std,
        transforms=transforms,
        multi_scale=multi_scale,
        draw_distribution=False
    )

    for fold_index, [train_loader, valid_loader, train_labels_number] in enumerate(zip(train_dataloaders, val_dataloaders, train_labels_number_folds)):
        if fold_index in config.selected_fold:
            train_val = TrainVal(config, fold_index, train_labels_number)
            train_val.train(train_loader, valid_loader)
