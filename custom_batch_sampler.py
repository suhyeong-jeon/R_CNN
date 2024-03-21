import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from custom_finetune_dataset import CustomFineTuneDataset

# Dataset은 idx로 데이터를 가져오도록 설계 되었다. 이 때 Sampler는 이 idx 값을 컨트롤하는 방법임
# 따라서 sampler를 사용할 때는 shuffle 파라미터는 False가 되어야함

class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:

        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive # 32
        self.batch_negative = batch_negative # 96

        length = num_positive + num_negative
        self.idx_list = list(range(length)) # [0, 1, 2, 3, ..., length-1]

        self.batch = batch_negative + batch_positive # mini batch = 128
        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                # length = num_positive + num_negative니 positive중 32개를 랜덤으로 뽑고 negative중 96개를 랜덤으로 뽑는다는 뜻
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive), # idx_list에서 num_positive 이하의 인덱스를
                                                                                        # 가진 요소들 중에서 batch_positive개의 요소를 랜덤으로 샘플링
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list) # tmp에는 양성 및 음성 샘플이 랜덤하게 섞여 있는 배열이 저장됨

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


def test():
    root_dir = './my_voc2007/finetune_car/train'
    train_data_set = CustomFineTuneDataset(root_dir)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())

    first_idx_list = list(train_sampler.__iter__())[:128]
    print(first_idx_list)

    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))


if __name__ == '__main__':
    test()