import pandas as pd
import torch
from PIL import Image
import os
from torchvision import transforms


curr_dir = os.path.dirname(__file__)
data_path = os.path.join(curr_dir, '..')


def list2dict(l: list):
    d = dict()
    id_num = 0
    for i in l:
        if i in d.keys():
            continue
        d[i] = id_num
        id_num += 1

    return d


class BatchLoader:
    def __init__(self, reso='256', train='train', mag='20x', size='6w', modality='texture', batch_size=32):
        assert mag in ['20x', '20x+50x', '50x']
        assert size in ['6w', '9w']
        assert modality in ['texture', 'heightmap']
        assert train in ['train', 'test']

        self.reso = reso
        self.train = train
        self.mag = mag
        self.size = size
        self.modality = modality
        self.batch_size = batch_size

        self.meta_data = pd.read_csv(os.path.join(data_path, 'LUWA', 'CSV',
                                                  f'{reso}_{mag}_{size}_{train}.csv'))
        picture_path = os.path.join(data_path,
                                    'LUWA',
                                    f'{self.reso}',
                                    f'{self.mag}',
                                    f'{self.modality}')

        self.image_list = self.meta_data.iloc[:, 0].to_list()
        self.label_list = self.meta_data.iloc[:, 1].to_list()
        self.mag_list = self.meta_data.iloc[:, 2].to_list()
        self.batch_idx = -1
        self.total_len = len(self.image_list) // batch_size + 1

        self.images = list()
        trans = transforms.ToTensor()
        for i in range(len(self.image_list)):
            image = Image.open(os.path.join(picture_path, f'{self.label_list[i]}'.lower(), self.image_list[i]))
            self.images.append(trans(image).unsqueeze(0))
        self.images = torch.cat(self.images, dim=0)

        labels_dict = list2dict(self.label_list)
        self.labels = torch.tensor([labels_dict[l] for l in self.label_list])

        shuffle_idx = torch.randperm(self.images.shape[0])
        self.images = self.images[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

        return

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_idx += 1

        if self.batch_idx >= self.total_len:
            self.batch_idx = -1
            raise StopIteration

        idx = range(self.batch_idx * self.batch_size, min((self.batch_idx + 1) * self.batch_size, self.images.shape[0]))
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.total_len


if __name__ == '__main__':
    batch_loader = BatchLoader()
