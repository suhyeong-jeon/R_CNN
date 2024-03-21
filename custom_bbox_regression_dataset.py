import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils.util as util


class BBoxRegressionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super(BBoxRegressionDataset, self).__init__()
        self.transform = transform

        samples = util.parse_car_csv(root_dir)
        jpeg_list = list()
        box_list = list()
        for i in range(len(samples)):
            sample_name = samples[i]

            jpeg_path = os.path.join(root_dir, 'JPEGImages', sample_name + '.jpg')
            bndbox_path = os.path.join(root_dir, 'bndboxs', sample_name + '.csv')
            positive_path = os.path.join(root_dir, 'positive', sample_name + '.csv')

            jpeg_list.append(cv2.imread(jpeg_path))
            bndboxes = np.loadtxt(bndbox_path, dtype=np.int64, delimiter=' ')
            positives = np.loadtxt(positive_path, dtype=np.int64, delimiter=' ')

            if len(positives.shape) == 1:
                bndbox = self.get_bndbox(bndboxes, positives)
                box_list.append({'image_id': i, 'positive': positives, 'bndbox': bndbox})
            else:
                for positive in positives:
                    bndbox = self.get_bndbox(bndboxes, positive)
                    box_list.append({'image_id': i, 'positive': positive, 'bndbox': bndbox})

        self.jpeg_list = jpeg_list
        self.box_list = box_list

    def __getitem__(self, index: int):
        assert index < self.__len__(), '数据集大小为%d，当前输入下标为%d' % (self.__len__(), index)

        box_dict = self.box_list[index]
        image_id = box_dict['image_id']
        positive = box_dict['positive']
        bndbox = box_dict['bndbox']

        jpeg_img = self.jpeg_list[image_id]
        xmin, ymin, xmax, ymax = positive
        image = jpeg_img[ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        target = dict()
        p_w = xmax - xmin
        p_h = ymax - ymin
        p_x = xmin + p_w / 2
        p_y = ymin + p_h / 2

        xmin, ymin, xmax, ymax = bndbox
        g_w = xmax - xmin
        g_h = ymax - ymin
        g_x = xmin + g_w / 2
        g_y = ymin + g_h / 2

        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_h
        t_w = np.log(g_w / p_w)
        t_h = np.log(g_h / p_h)

        return image, np.array((t_x, t_y, t_w, t_h))

    def __len__(self):
        return len(self.box_list)

    def get_bndbox(self, bndboxes, positive):

        if len(bndboxes.shape) == 1:
            return bndboxes
        else:
            scores = util.iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]


def test():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = './my_voc2007/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)

    print(data_set.__len__())
    image, target = data_set.__getitem__(10)
    print(image.shape)
    print(target)
    print(target.dtype)


def test2():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_root_dir = './my_voc2007/bbox_regression'
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

    items = next(data_loader.__iter__())
    datas, targets = items
    print(datas.shape)
    print(targets.shape)
    print(targets.dtype)


if __name__ == '__main__':
    test()
    # test2()