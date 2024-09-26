import pandas as pd
import torch
import tqdm
from PIL import Image
import os
from torchvision import transforms


curr_dir = os.path.dirname(__file__)
data_path = os.path.join(curr_dir, '..')
torch.manual_seed(1234)


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
    transform = None

    def __init__(self, reso='256', train='train', mag='20x', size='6w', modality='texture', batch_size=32,
                 shuffle=False):
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
        self.picture_path = os.path.join(data_path,
                                         'LUWA',
                                         f'{self.reso}',
                                         f'{self.mag}',
                                         f'{self.modality}')

        self.image_list = self.meta_data.iloc[:, 0].to_list()
        self.label_list = self.meta_data.iloc[:, 1].to_list()
        self.mag_list = self.meta_data.iloc[:, 2].to_list()
        self.batch_idx = -1
        self.total_len = len(self.image_list) // batch_size + 1

        # shuffle the data
        if shuffle:
            shuffle_idx = torch.randperm(len(self.image_list))
            self.image_list = [self.image_list[idx] for idx in shuffle_idx]
            self.label_list = [self.label_list[idx] for idx in shuffle_idx]
        # print(idx, self.label_list[idx])

        # labels_dict = list2dict(self.label_list)
        labels_dict = {'ANTLER': 0, 'BEECHWOOD': 1, 'BEFOREUSE': 2, 'BONE': 3, 'IVORY': 4, 'SPRUCEWOOD': 5}
        self.labels = torch.tensor([labels_dict[l] for l in self.label_list])

        self.class_num = len(labels_dict.keys())

        return

    def get_statistics(self):
        means = torch.zeros([3])
        stds = torch.zeros([3])

        pbar = tqdm.tqdm(zip(self.image_list, self.label_list), total=len(self.image_list))

        for image_name, label_name in pbar:
            image = Image.open(os.path.join(self.picture_path, f'{label_name}'.lower(), image_name))
            image = transforms.ToTensor()(image)
            means += torch.mean(image, dim=(-1, -2))
            stds += torch.std(image, dim=(-1, -2))

        means /= len(self.image_list)
        stds /= len(self.image_list)

        return means, stds

    def get_batch(self, batch_idx):
        indices = range(batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, len(self.image_list)))

        labels = self.labels[indices]
        image_names = [self.image_list[idx] for idx in indices]

        images = list()
        trans = transforms.ToTensor()
        for i in range(len(image_names)):
            image = Image.open(os.path.join(self.picture_path,
                                            f'{self.label_list[batch_idx * self.batch_size + i]}'.lower(),
                                            image_names[i]))
            image = trans(image).unsqueeze(0)
            images.append(image)

        images = torch.cat(images, dim=0)

        return images, labels

    def prepare_transform(self, means, stds, mode='train'):
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize([224, 224], antialias=True),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                # transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
            ])

        return

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_idx += 1

        if self.batch_idx >= self.total_len:
            raise StopIteration

        return self.get_batch(self.batch_idx)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label_name = self.label_list[idx]
        label = self.labels[idx]

        image = Image.open(os.path.join(self.picture_path, f'{label_name}'.lower(), image_name))
        image = transforms.ToTensor()(image)
        if self.transform is not None:
            image = self.transform(image)
        # print(idx, '\t', label, '\n')
        # print(label)
        return image, label


if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader
    batch_loader = BatchLoader(train='test')
    data_loader = DataLoader(batch_loader, batch_size=4, num_workers=4)
    for im, la in data_loader:
        print(im.shape, la.shape)
        break
