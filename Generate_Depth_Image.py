import datetime as dt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from skimage.io import imread, imsave
import cv2
import os
import pandas as pd
from api import PRN
import utils123.depth_image as DepthImage
from kutils.image.face_ops import extract_face
from data_ingestion.transforms.basic_transformations import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

prn = PRN(is_dlib=True, is_opencv=False)

final_path = r"B:\PycharmProjects\spoof-detection\youtube1\vids1\depht_map"
df = pd.read_csv(
    r"B:\PycharmProjects\spoof-detection\youtube1\all_images.txt",
    sep=" ",
    names=["path", "target"],
)

df1 = pd.DataFrame(columns=["path", "target"])


paths = df["path"]
labels = df["target"]
# paths =df.loc[df["target"] == 0]["path"]
# paths =df.loc[df["target"] == 0]["path"]
# labels=df.loc[df["target"] == 0]["target"]

count = 0
for path_image, label in zip(paths, labels):
    print(count)

    try:
        image = imread(path_image)
        image = extract_face(image, 1)[0]
        image_shape = [image.shape[0], image.shape[1]]

        #%%
        pos = prn.process(image, None, None, image_shape)

        kpt = prn.get_landmarks(pos)

        # 3D vertices
        vertices = prn.get_vertices(pos)

        depth_scene_map = DepthImage.generate_depth_image(
            vertices, kpt, image.shape, isMedFilter=True
        )
        output_path = os.path.join(final_path, path_image.split("/")[-1])
        imsave(output_path, depth_scene_map)
        image = resize(image, 256, 256)
        df1.loc[count] = [path_image.replace("\\", "/"), label]
        count += 1
    except Exception as e:

        print(e)
        print(path_image)
#%%
# df1=df[:443]
#%%
df1.to_csv(
    r"B:\PycharmProjects\spoof-detection\youtube1\vids1\all_image_deph_map.txt",
    sep=" ",
    index=None,
    header=None,
)
