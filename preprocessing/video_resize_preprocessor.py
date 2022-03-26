# import the necessary packages
import cv2
from pathlib import Path
import os
import numpy as np
from skimage.transform import resize

class VideoResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA, verbose=500):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.verbose = verbose

    def preprocess_and_save(self, vid_path, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        parent = Path(vid_path).parent.parent
        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '_' + str(self.width) + 'x' + str(self.height))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        fps = video.get(cv2.CAP_PROP_FPS)
        out_vid_path = out_dir / video_name

        # verify if the resized video alredy exists, if it does, then proceed to the next video
        if os.path.isfile(out_vid_path):
            out_vid = cv2.VideoCapture(str(out_vid_path))
            if int(out_vid.get(cv2.CAP_PROP_FRAME_COUNT)) == vid_len:
                return None, None, out_vid_path
                # return np.load(out_vid_path, mmap_mode='r'), None, out_vid_path
            else:
                out_vid.release()

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_vid = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (self.width, self.height))

        i = 0

        while video.isOpened():
            # show an update every 'verbose' images
            if self.verbose > 0 and i > 0 and (i + 1) % self.verbose == 0:
                print("[INFO] frame resizing processed {}/{}".format(i + 1, vid_len))

            i += 1
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=self.inter)
            out_vid.write(frame)

        out_vid.release()

        return None, None, out_vid_path

    def preprocess_and_save_npy(self, vid_path, *_):
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent
        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '_' + str(self.width) + 'x' + str(self.height))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_vid_path = out_dir / (str(video_name).split('.')[0] + '.npy')

        video = cv2.VideoCapture(str(vid_path))
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # verify if the resized video alredy exists, if it does, then proceed to the next video
        if os.path.isfile(out_vid_path):
            out_vid = np.load(out_vid_path, mmap_mode='r')
            if out_vid.shape[0] == vid_len:
                return None, None, out_vid_path
                # return np.load(out_vid_path, mmap_mode='r'), None, out_vid_path
        frames = []
        for i in range(100):
            ret, frame = video.read()
            frames.append(cv2.resize(frame, (self.width, self.height), interpolation=self.inter))

        video.release()
        np.save(out_vid_path, np.array(frames))
        return None, None, out_vid_path

    def preprocess(self, vid_path, video, *_):
        frames = []
        frames = [cv2.resize(frame, (self.width, self.height), interpolation=self.inter) for frame in video]
        frames = np.array(frames)
        return frames, None, vid_path
