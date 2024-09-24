import pandas as pd
from PIL import Image
import os


curr_dir = os.path.dirname(__file__)
data_path = os.path.join(curr_dir, '..')


class BatchLoader:
    def __init__(self, reso, train=True, mag='20x', size='6w', modality='texture', batch_size=32):
        assert mag in ['20x', '20x+50x', '50x']
        assert size in ['6w', '9w']
        assert modality in ['texture', 'heightmap']

        self.reso = reso
        self.train = train
        self.mag = mag
        self.size = size
        self.modality = modality
        self.batch_size = batch_size

        self.meta_data = pd.read_csv(f'./LUWA/CSV{reso}_{mag}_{size}_train.csv')

        self.image_list = self.meta_data.iloc[:, 0].to_list()
        self.label_list = self.meta_data.iloc[:, 1].to_list()
        self.mag_list = self.meta_data.iloc[:, 2].to_list()
        self.idx = 0
        self.total_len = len(self.image_list) // batch_size + 1

    def get_batch(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        image_names = self.image_list[self.idx * self.batch_size:
                                      min((self.idx + 1) * self.batch_size, self.total_len)]
        images = list()
        for _ in range(min((self.idx + 1) * self.batch_size, self.total_len) - self.idx * self.batch_size):
            images.append(Image.open(os.path.join(data_path,
                                                  f'{self.reso}',
                                                  f'{self.mag}',
                                                  f'{self.modality}',
                                                  f'{}')))

    def __len__(self):
        return self.total_len

resolution = 256
config = pd.read_csv('./LUWA/CSV/256_50x_6w_train.csv')
image_name = config.iloc[:, 0]
image_name = image_name.to_list()
print(len(image_name))
