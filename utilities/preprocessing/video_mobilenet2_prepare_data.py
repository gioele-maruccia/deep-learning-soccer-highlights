# import the necessary packages
import cv2
from pathlib import Path
import os
import subprocess
from subprocess import STDOUT, DEVNULL
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from skimage.transform import resize


class VideoMobilenet2PrepareData:
    def __init__(self, verbose=500):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.verbose = verbose

    def preprocess_and_save(self, vid_path, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_data_preparation')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        if os.path.exists(out_vid_path):
            return None, None, out_vid_path
        count = 0
        # print('[INFO] Extracting frames from video: ', vid_path)
        vidcap = cv2.VideoCapture(vid_path)
        features = []
        success = True
        while success:
            #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                img = image
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                feature = preprocess_input(input)
                features.append(feature)
                count = count + 1

        unscaled_features = np.array(features)
        # np.save(out_vid_path, unscaled_features)

        # print("out_vid_path = ", out_vid_path)
        # print("unscaled_features shape", unscaled_features.shape)
        return unscaled_features, None, out_vid_path

    def preprocess(self, vid_path, video, *_):
        count = 0
        # print('[INFO] Extracting frames from video: ', vid_path)
        features = []
        success = True
        for i in range(video.shape[0]):
            # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            image = video[i]
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                features.append(input)
                count = count + 1

        unscaled_features = np.array(features)
        # print("out_vid_path = ", out_vid_path)
        # print("unscaled_features shape", unscaled_features.shape)
        return unscaled_features, None, None
        # return None, None, out_vid_path
