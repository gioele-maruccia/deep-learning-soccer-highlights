# import the necessary packages
import cv2
from pathlib import Path
import os
import subprocess
from subprocess import STDOUT, DEVNULL
class VideoColorPreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA, verbose=500):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        self.verbose = verbose

    def preprocess(self, vid_path, *_):
        # resize the video to a fixed size, ignoring the aspect
        # ration
        video = cv2.VideoCapture(vid_path)
        video_name = Path(vid_path.split(os.path.sep)[-1])
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        parent = Path(vid_path).parent.parent
        out_dir = parent / (vid_path.split(os.path.sep)[-2] + '__grayscale')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        fps = video.get(cv2.CAP_PROP_FPS)
        out_vid_path = out_dir / video_name

        # verify if the resized video alredy exists, if it does, then proceed to the next video
        if os.path.isfile(out_vid_path):
            out_vid = cv2.VideoCapture(str(out_vid_path))
            if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == vid_len:
                return None, None, out_vid_path

        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out_vid = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (self.width, self.height))

        sp = subprocess.call(['ffmpeg', '-i', vid_path, '-vf', 'hue=s=0', str(out_vid_path)],  stdout=DEVNULL, stderr=STDOUT)

        return None, None, out_vid_path
