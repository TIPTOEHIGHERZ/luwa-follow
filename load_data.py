import pandas as pd


config = pd.read_csv('./LUWA/CSV/256_50x_6w_train.csv')
image_name = config.iloc[:, 0]
image_name = image_name.to_list()
print(len(image_name))
