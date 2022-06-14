import datetime as dt
import numpy as np
import scipy.io as sio

# import tensorflow as tf
# from PRNet.cv_plot import plot_kpt, plot_vertices, plot_pose_box
import matplotlib.pyplot as plt
from utils_pose.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from utils_pose.estimate_pose import estimate_pose
from skimage.io import imread, imsave
import cv2
import os
import pandas as pd
from api import PRN
import utils123.depth_image as DepthImage
from kutils.image.face_ops import extract_face
from data_ingestion.transforms.basic_transformations import resize
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

prn = PRN(is_dlib=True, is_opencv=False)

path = r"/PRNet_Depth_Generation\angles"

count = 0
#%%
def pose_full_image(image):

    # image = extract_face(image, 1)[0]
    image_shape = [image.shape[0], image.shape[1]]

    pos = prn.process(image, None, None, image_shape)

    kpt = prn.get_landmarks(pos)

    # 3D vertices
    vertices = prn.get_vertices(pos)
    camera_matrix, pose = estimate_pose(vertices)
    plot = plot_pose_box(image, camera_matrix, kpt)
    return plot, pose


def pose_croped_image(image):

    image = extract_face(image, 1)[0]
    image_shape = [image.shape[0], image.shape[1]]

    pos = prn.process(image, None, None, image_shape)

    kpt = prn.get_landmarks(pos)

    # 3D vertices
    vertices = prn.get_vertices(pos)
    camera_matrix, pose = estimate_pose(vertices)
    plot = plot_pose_box(image, camera_matrix, kpt)
    return plot, pose


#%%
wandb.init(project="Spoof detection", name="Tensorflow PRnet test resize")
table = wandb.Table(
    columns=["Original", "Pose", "Cropped pose", "Angle", "Cropped angle"]
)

for i, path_image in enumerate(os.listdir(path)):
    print(i)

    path_image = os.path.join(path, path_image)
    image = imread(path_image)
    # image = cv2.resize(image, (256, 256))
    original = wandb.Image(image)

    plot, pose = pose_full_image(image)
    try:
        plot_C, pose_C = pose_croped_image(image)

    except:
        plot_C, pose_C = plot, pose
    table.add_data(
        original,
        wandb.Image(plot),
        wandb.Image(plot_C),
        list(pose),
        list(pose_C),
    )
wandb.log({"Analisis": table})
wandb.finish()
