import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.util import parse_car_csv


"""
모델을 학습시키기 위한 대상을 정의
"""


class CustomFineTuneDataset(Dataset):
    # 2000개의 region proposal positive와 negative에 대해서 rects와 region의 개수인 size를 리스트에 저장
    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)

        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")) for sample_name in samples]

        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + "_1.csv") for sample_name in samples]
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + "_0.csv") for sample_name in samples]

        positive_sizes = list() # [1, .....]
        negative_sizes = list()

        positive_rects = list() # [(x, y, w, h), ....]
        negative_rects = list()

        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int64, delimiter=' ')

            if len(rects.shape) == 1: # csv file의 데이터가 1줄밖에 없다면
                if rects.shape[0] == 4:
                    positive_rects.append(rects)
                    positive_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))

        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int64, delimiter = ' ')

            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    positive_sizes.append(0)

            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))
    
    # index에 대한 2000개의 positive, negative region proposal의 4개의 좌표를 받고 원본 image에서 4개의 좌표를 사용해 crop한
    # 이미지를 transformed하여 target과 함께 return함
    def __getitem__(self, index):
        image_id = len(self.jpeg_images) - 1

        if index < self.total_positive_num:
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects[index]

            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i+1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax] # selective search를 통해 도출한 Region proposal을 image에 추출

        else:
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]

            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= idx < np.sum(self.negative_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

            # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
            #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self) -> int:
        return self.total_positive_num

    def get_negative_num(self) -> int:
        return self.total_negative_num

def test(idx):
    root_dir = './my_voc2007/finetune_car/train'
    train_data_set = CustomFineTuneDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    image, target = train_data_set.__getitem__(idx)
    print('target: %d' % target)

    image = Image.fromarray(image)
    print(image)
    print(type(image))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    test(24768)