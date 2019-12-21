import torch
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
plt.switch_backend('agg')
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import collections


class TrainDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std, transforms=None, choose_dataset='combine', multi_scale=False):
        """
        Args:
            data_root: str, 数据集根目录
            sample_list: list, 样本名
            label_list: list, 类标, 与sample_list中的样本按照顺序对应
            size: [height, width], 图片的目标大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            transforms: callable, 数据集转换方式
            choose_dataset: str，选择什么数据集
            multi_scale: bool, 是否使用多尺度训练
        """
        super(TrainDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        self.choose_dataset = choose_dataset

        if self.choose_dataset == 'combine':
            pass
        else:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if self.choose_dataset == 'only_self':
                    if 'img' in sample:
                        sample_list.append(sample)
                        label_list.append(label)
                elif self.choose_dataset == 'only_official':
                    if 'img' not in sample:
                        sample_list.append(sample)
                        label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list        
        
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
        self.multi_scale = multi_scale
    
    def __getitem__(self, index):
        """
        Args:
            index: int, 当前的索引下标

        Returns:
            image_name: str；图片名称
            image: [channel, height, width] tensor, 当前索引下标对应的图像数据
            label: [1] tensor, 当前索引下标对应的图像数据对应的类标
        """
        image_name = self.sample_list[index]
        sample_path = os.path.join(self.data_root, image_name)
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        if self.transforms:
            image = np.asarray(image)
            image = self.transforms(image)
            image = Image.fromarray(image)
        
        # 如果不进行多尺度训练，则将图片转换为指定的图片大小，并转换为tensor
        if self.multi_scale:
            image = T.Resize(self.size, interpolation=3)(image)
            image = np.asarray(image)
        else:
            transform_train_list = [
                        T.Resize(self.size, interpolation=3),
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std)
                    ]          
            transform_compose = T.Compose(transform_train_list)
            image = transform_compose(image)
        label = torch.tensor(label).long()

        return image_name, image, label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)
    

class ValDataset(Dataset):
    def __init__(self, data_root, sample_list, label_list, size, mean, std, choose_dataset='combine', multi_scale=False):
        """
        Args:
            data_root: str, 数据集根目录
            sample_list: list, 样本名
            label_list: list, 类标, 与sample_list中的样本按照顺序对应
            size: [height, width], 图片的目标大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            choose_dataset: str，选择什么数据集
            multi_scale: bool, 是否使用多尺度训练
        """
        super(ValDataset, self).__init__()
        self.data_root = data_root
        self.sample_list = sample_list
        self.label_list = label_list
        self.choose_dataset = choose_dataset

        if self.choose_dataset == 'combine':
            pass
        else:
            sample_list = []
            label_list = []
            for sample, label in zip(self.sample_list, self.label_list):
                if self.choose_dataset == 'only_self':
                    if 'img' in sample:
                        sample_list.append(sample)
                        label_list.append(label)
                elif self.choose_dataset == 'only_official':
                    if 'img' not in sample:
                        sample_list.append(sample)
                        label_list.append(label)
            self.sample_list = sample_list
            self.label_list = label_list        

        self.size = size
        self.mean = mean
        self.std = std
        self.multi_scale = multi_scale
    
    def __getitem__(self, index):
        """
        Args:
            index: int, 当前的索引下标

        Returns:
            image_name: str；图片名称
            image: [channel, height, width] tensor, 当前索引下标对应的图像数据
            label: [1] tensor, 当前索引下标对应的图像数据对应的类标
        """
        image_name = self.sample_list[index]
        sample_path = os.path.join(self.data_root, image_name)
        image = Image.open(sample_path).convert('RGB')
        label = self.label_list[index]
        
        if self.multi_scale:
            image = T.Resize(self.size, interpolation=3)(image)
        else:
            transform_val_list = [ 
                        T.Resize(self.size, interpolation=3),
                        T.ToTensor(),
                        T.Normalize(self.mean, self.std)
                    ]          
            transform_compose = T.Compose(transform_val_list)
            image = transform_compose(image)
        label = torch.tensor(label).long()

        return image_name, image, label

    def __len__(self):
        """ 得到训练数据集总共有多少个样本
        """
        return len(self.sample_list)


class GetDataloader(object):
    def __init__(
        self, 
        data_root, 
        folds_split=1, 
        test_size=None, 
        label_names_path='data/huawei_data/label_id_name.json', 
        choose_dataset='combine',
        load_split_from_file=None
        ):
        """
        Args:
            data_root: str, 数据集根目录
            folds_split: int, 划分为几折，当划分1折时，则根据test_size划分验证集与训练集
            test_size: 验证集占的比例, [0, 1]
            label_names_path: str, label_id_name.json的路径
            choose_dataset: str，选择什么数据集
            load_split_from_file: str, 存放数据集划分的文件的路径，如果存在则从文件加载，否则在线生成
        """
        self.data_root = data_root
        self.folds_split = folds_split
        self.samples, self.labels = self.get_samples_labels()
        self.test_size = test_size
        self.choose_dataset = choose_dataset
        self.load_split_from_file = load_split_from_file
        with open(label_names_path, 'r') as f:
            self.label_to_name = json.load(f)
            self.name_to_label = {v: k for k, v in self.label_to_name.items()}

        if folds_split == 1:
            if not test_size:
                raise ValueError('You must specified test_size when folds_split equal to 1.')
    
    def get_dataloader(self, batch_size, image_size, mean, std, transforms=None, multi_scale=False, draw_distribution=True):
        """得到数据加载器
        Args:
            batch_size: int, 批量大小
            image_size: [height, width], 图片大小
            mean: tuple, 通道均值
            std: tuple, 通道方差
            transforms: callable, 数据增强方式
            multi_scale: bool, 是否使用多尺度训练
            draw_distribution: bool, 是否画出分布图
        Return:
            train_dataloader_folds: list, [train_dataloader_0, train_dataloader_1,...]
            valid_dataloader_folds: list, [val_dataloader_0, val_dataloader_1, ...]
            train_labels_number_fold: list, list中每一个数据均为list类型，表示某一折的[number_class0, number__class1, ...]
            val_labels_number_folds: list, list中每一个数据均为list类型，表示某一折的[number_class0, number__class1, ...]
        """
        train_lists, val_lists = self.get_split()
        train_dataloader_folds, valid_dataloader_folds = list(), list()
        train_labels_number_folds, val_labels_number_folds = self.draw_train_val_distribution(train_lists, val_lists, draw_distribution)

        for train_list, val_list in zip(train_lists, val_lists):
            train_dataset = TrainDataset(
                self.data_root, 
                train_list[0], 
                train_list[1], 
                image_size,
                transforms=transforms, 
                mean=mean, 
                std=std, 
                choose_dataset=self.choose_dataset, 
                multi_scale=multi_scale
                )
            # 默认不在验证集上进行多尺度
            val_dataset = ValDataset(
                self.data_root, 
                val_list[0], 
                val_list[1], 
                image_size, 
                mean=mean, 
                std=std, 
                choose_dataset=self.choose_dataset, 
                multi_scale=False
                )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=False
            )
            train_dataloader_folds.append(train_dataloader)
            valid_dataloader_folds.append(val_dataloader)
        return train_dataloader_folds, valid_dataloader_folds, train_labels_number_folds, val_labels_number_folds

    def draw_train_val_distribution(self, train_lists, val_lists, draw_distribution):
        """ 画出各个折的训练集与验证集的数据分布

        Args:
            train_lists: list, 每一个数据均为[train_sample, train_label], train_sample: list, 样本名称， train_label: list, 样本类标
            val_lists: list, 每一个数据均为[val_sample, val_label]， val_sample: list, 样本名称， val_label: list, 样本类标
            draw_distribution: bool, 是否画出分布图
        Returns:
            train_labels_number_fold: list, list中每一个数据均为list类型，表示某一折的[number_class0, number__class1, ...]
            val_labels_number_folds: list, list中每一个数据均为list类型，表示某一折的[number_class0, number__class1, ...]
        """
        train_labels_number_folds, val_labels_number_folds = [], []
        for index, (train_list, val_list) in enumerate(zip(train_lists, val_lists)):
            train_labels_number = {}
            for label in train_list[1]:
                if label in train_labels_number.keys():
                    train_labels_number[label] += 1
                else:
                    train_labels_number[label] = 1
            if draw_distribution:
                self.draw_labels_number(train_labels_number, phase='Train_%s' % index)
            train_labels_number_folds.append(list(collections.OrderedDict(sorted(train_labels_number.items())).values()))

            val_labels_number = {}
            for label in val_list[1]:
                if label in val_labels_number.keys():
                    val_labels_number[label] += 1
                else:
                    val_labels_number[label] = 1
            if draw_distribution:
                self.draw_labels_number(val_labels_number, phase='Val_%s' % index)
            val_labels_number_folds.append(list(collections.OrderedDict(sorted(val_labels_number.items())).values()))
        return train_labels_number_folds, val_labels_number_folds

    def draw_labels_number(self, labels_number, phase='Train'):
        """ 画图函数
        Args:
            labels_number: dict, {label_1: number_1, label_2: number_2, ...}
            phase: str, 当前模式
        """
        labels_number = {k: v for k, v in sorted(labels_number.items(), key=lambda item: item[1])}
        labels = labels_number.keys()
        number = labels_number.values()
        name = [self.label_to_name[str(label)] for label in labels]
        
        plt.figure(figsize=(20, 16), dpi=240)
        font = FontProperties(fname=r"font/simhei.ttf", size=7)
        ax1 = plt.subplot(111)
        x_axis = range(len(labels))
        rects = ax1.bar(x=x_axis, height=number, width=0.8, label='Label Number')
        plt.ylabel('Number')
        plt.xticks([index + 0.13 for index in x_axis], name, fontproperties=font, rotation=270)
        plt.xlabel('Labels')
        plt.title('%s: Sample Number of Each Label' % phase)
        plt.legend()

        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
        plt.savefig('readme/%s.jpg' % phase, dpi=240)
        
    def get_split(self):
        """对数据集进行划分
        Return:
            train_list: list, 每一个数据均为[train_sample, train_label], train_sample: list, 样本名称， train_label: list, 样本类标
            val_list: list, 每一个数据均为[val_sample, val_label]， val_sample: list, 样本名称， val_label: list, 样本类标
        """
        if self.load_split_from_file:
            print('Loading dataset split from %s' % self.load_split_from_file)
            with open(self.load_split_from_file, 'r') as f:
                train_list, val_list = json.load(f)
        else:
            if self.folds_split == 1:
                train_list, val_list = self.get_data_split_single()
            else:
                train_list, val_list = self.get_data_split_folds()

        return train_list, val_list
        
    def get_data_split_single(self):
        """随机划分训练集和验证集
        Return:
            [train_samples, train_labels], train_samples: list, 样本名称， train_labels: list, 样本类标
            [val_samples, val_labels], val_samples: list, 样本名称， val_labels: list, 样本类标
        """
        samples_index = [i for i in range(len(self.samples))]
        train_index, val_index = train_test_split(samples_index, test_size=self.test_size, random_state=69)
        train_samples = [self.samples[i] for i in train_index]
        train_labels = [self.labels[i] for i in train_index]
        val_samples = [self.samples[i] for i in val_index]
        val_labels = [self.labels[i] for i in val_index]
        return [[train_samples, train_labels]], [[val_samples, val_labels]]
    
    def get_data_split_folds(self):
        """交叉验证的数据划分
        Return:
            train_folds: list, 所有折的[train_samples, train_labels], train_samples: list, 样本名称， train_labels: list, 样本类标
            val_folds: list, 所有折的[val_samples, val_labels], val_samples: list, 样本名称， val_labels: list, 样本类标
        """
        skf = StratifiedKFold(n_splits=self.folds_split, shuffle=True, random_state=69)
        train_folds = []
        val_folds = []
        for train_index, val_index in skf.split(self.samples, self.labels):
            train_samples = ([self.samples[i] for i in train_index])
            train_labels = ([self.labels[i] for i in train_index])
            val_samples = ([self.samples[i] for i in val_index])
            val_labels = ([self.labels[i] for i in val_index])
            train_folds.append([train_samples, train_labels])
            val_folds.append([val_samples, val_labels])
        return train_folds, val_folds

    def get_samples_labels(self):
        """ 得到所有的图片名称以及对应的类标
        Returns:
            samples: list, 所有的图片名称
            labels: list, 所有的图片对应的类标, 和samples一一对应
        """
        files_list = sorted(os.listdir(self.data_root))
        # 过滤得到标注文件
        annotations_files_list = [f for f in files_list if f.split('.')[1] == 'txt']

        samples = []
        labels = []
        for annotation_file in annotations_files_list:
            annotation_file_path = os.path.join(self.data_root, annotation_file)
            with open(annotation_file_path, encoding='utf-8-sig') as f:
                for sample_label in f:
                    try:
                        sample_name = sample_label.split(', ')[0]
                        label = int(sample_label.split(', ')[1])
                    except:
                        sample_name = sample_label.split(',')[0]
                        label = int(sample_label.split(',')[1])
                    samples.append(sample_name)
                    labels.append(label)
        return samples, labels


def multi_scale_transforms(image_size, images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ TrainDataset类中无法实现可变的图像尺寸，借助这个函数实现

    Args:
        image_size: tuple, 图片的目标大小[height, width]
        images: tensor, 一个batch的数据，尺度为[batch_size, height, width, 3]
        mean: tuple, 通道均值
        std: tuple, 通道方差

    Returns:
        images_resize: tensor, 一个batch的数据，尺度为[batch_size, 3, height, width]
    """
    transform_train_list = [
                T.Resize(image_size, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]
    transform_compose = T.Compose(transform_train_list)
    images = images.numpy()
    images_resize = torch.zeros(images.shape[0], 3, image_size[0], image_size[1])
    for index in range(images.shape[0]):
        image = transform_compose(Image.fromarray(images[index]))
        images_resize[index] = image

    return images_resize
    

if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    folds_split = 1
    test_size = 0.2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    get_dataloader = GetDataloader(data_root, folds_split=1, test_size=test_size)
    train_list, val_list = get_dataloader.get_split()
    train_dataset = TrainDataset(data_root, train_list[0], train_list[1], size=[224, 224], mean=mean, std=std)
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
