import torch.nn as nn
from torch.utils.data import Dataset
from custom_classifier_dataset import CustomClassifierDataset

# Hard negative mining은 모델이 예측에 실패하는 어려운 sample들을 모으는 기법.
# Hard negative mining을 통해 수집된 데이터를 활용해 모델을 강하게 학습시키는 것이 가능해짐.
# False Positive Sample을 학습 과정에 추가하여 재학습시켜 False Positive라고 판단하는 오류를 줄임.

class CustomHardNegativeMiningDataset(Dataset):

    def __init__(self, negative_list, jpeg_images, transform=None):
        self.negative_list = negative_list
        self.jpeg_images = jpeg_images
        self.transform = transform

    def __getitem__(self, index: int):
        target = 0

        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']
        image_id = negative_dict['image_id']

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        if self.transform:
            image = self.transform(image)

        return image, target, negative_dict

    def __len__(self) -> int:
        return len(self.negative_list)


if __name__ == '__main__':
    root_dir = './my_voc2007/classifier_car/train'
    data_set = CustomClassifierDataset(root_dir)

    negative_list = data_set.get_negatives()
    jpeg_images = data_set.get_jpeg_images()
    transform = data_set.get_transform()

    hard_negative_dataset = CustomHardNegativeMiningDataset(negative_list, jpeg_images, transform=transform)
    image, target, negative_dict = hard_negative_dataset.__getitem__(100)

    print(image.shape)
    print(target)
    print(negative_dict)