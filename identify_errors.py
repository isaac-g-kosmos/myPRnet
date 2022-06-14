import utils123.depth_image as DepthImage
from kutils.image.face_ops import extract_face
from data_ingestion.transforms.basic_transformations import (
    resize,
    normalize_depth,
)
import pandas as pd
import numpy as np
from PIL import Image
from skimage.io import imread, imsave

final_path = "B:/PycharmProjects/spoof-detection/NUAA/images/mapping_cut"
missing_labels = pd.DataFrame(columns=["path", "label"])
df = pd.read_csv(
    "B:\\PycharmProjects\\spoof-detection\\NUAA\\images\\all_image_deph_map2.txt",
    sep=" ",
    names=["path", "target"],
)
#%%
paths = df["path"]

#%%
count = 0
caunt1 = 0
for path_image, target in zip(paths[5100:], df["target"][5100:]):
    try:
        print(caunt1)
        image = Image.open(path_image)
        image = np.array(image)
        image = extract_face(image, 1)[0]
        image = resize(image, 256, 256)
        image = normalize_depth(image)

        caunt1 += 1

    except:
        print(path_image)
        missing_labels.loc[count] = [path_image, target]
        count += 1
#%%
# df=pd.read_csv('B:\\PycharmProjects\\spoof-detection\\NUAA\\images\\all_image_deph_map2.txt',sep=' ',names=['path','target'])
df2 = df[~df.path.isin(missing_labels.path)]
#%%
df2.to_csv(
    "B:\\PycharmProjects\\spoof-detection\\NUAA\\images\\all_image_deph_map3.txt",
    sep=" ",
    columns=None,
    index=None,
    header=False,
)
#%%
import torch
import matplotlib.pyplot as plt

df = pd.read_csv(
    "B:\\PycharmProjects\\spoof-detection\\NUAA\\images\\all_image_deph_map2.txt",
    sep=" ",
    names=["path", "target"],
)
array = []
for path in df["path"][:10]:
    image = Image.open(path)
    print("hey")
    image = np.array(image)
    image = extract_face(image, 1)[0]
    image = resize(image, 256, 256)
    image = normalize_depth(image)
    array.append(image)

array = np.array(array)
#%%
tensor = torch.from_numpy(array)
tensor = tensor.detach().numpy()

plt.imshow(tensor[0])
plt.show()
