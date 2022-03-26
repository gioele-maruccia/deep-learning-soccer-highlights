# import the necessary packages
import cv2
from pathlib import Path
import os
import subprocess
from subprocess import STDOUT, DEVNULL
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import GlobalAveragePooling2D

class VideoVGG16FeatureExtractor:
    def __init__(self, model=None, verbose=500):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.verbose = verbose
        if model is None:
            self.model = Sequential()
            self.model.add(VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))
            self.model.add(Flatten())
            # self.model.add(GlobalAveragePooling2D())
            # self.model.add(Dense(1000)) #GIOXX
            self.model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def preprocess_and_save(self, vid_path, video, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_features_no_top_2')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        if os.path.exists(out_vid_path):
            # return None, None, out_vid_path
            x = np.load(out_vid_path, mmap_mode='r')
            return x, None, out_vid_path
        count = 0
        # print('[INFO] Extracting frames from video: ', vid_path)
        vidcap = cv2.VideoCapture(vid_path)
        features = []
        success = True
        while success:
            # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                #img = image
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = self.model.predict(input).ravel()
                features.append(feature)
                count = count + 1

        unscaled_features = np.array(features)
        np.save(out_vid_path, unscaled_features)
        # print("out_vid_path = ", out_vid_path)
        # print("unscaled_features shape", unscaled_features.shape)
        # return unscaled_features, None, out_vid_path
        return None, None, out_vid_path

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
                img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = self.model.predict(input).ravel()
                features.append(feature)
                count = count + 1

        unscaled_features = np.array(features)
        # print("out_vid_path = ", out_vid_path)
        # print("unscaled_features shape", unscaled_features.shape)
        return unscaled_features, None, None
        # return None, None, out_vid_path

    def preprocess_mod(self, vid_path, video, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent

        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__vgg16_features_no_top_2')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        if os.path.exists(out_vid_path):
            # return None, None, out_vid_path
            return np.load(out_vid_path, mmap_mode='r'), None, out_vid_path
        count = 0
        # print('[INFO] Extracting frames from video: ', vid_path)
        vidcap = cv2.VideoCapture(vid_path)
        features = []
        success = True
        while success:
            # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            if success:
                # img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                img = image
                input = img_to_array(img)
                input = np.expand_dims(input, axis=0)
                input = preprocess_input(input)
                feature = self.model.predict(input).ravel()
                features.append(feature)
                count = count + 1

        unscaled_features = np.array(features)
        #np.save(out_vid_path, unscaled_features)
        # print("out_vid_path = ", out_vid_path)
        # print("unscaled_features shape", unscaled_features.shape)
        # return unscaled_features, None, out_vid_path
        return unscaled_features, None, out_vid_path
