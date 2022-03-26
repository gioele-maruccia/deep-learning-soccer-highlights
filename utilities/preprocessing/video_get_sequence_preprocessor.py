# import the necessary packages
import cv2
from pathlib import Path
import os
import numpy as np

class VideoGetSequencePreprocessor:
    def __init__(self, verbose=500, clip_size=100):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.verbose = verbose
        self.clip_size = clip_size

    def preprocess(self, vid_path, video, isHighlight, isCelebration, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration):
        inCloud = False

        substr_start = vid_path.find("__")+2
        highlight_path = vid_path[substr_start::]

        if inCloud:
            parent = Path(vid_path).parent.parent
        else:
            parent = Path(vid_path).parent.parent.parent

        highlight_path = parent / "full_game_goal_cuts" / highlight_path

        off_frames = 550

        start = off_frames + initSeqFrame
        stop = start+self.clip_size

        video = cv2.VideoCapture(str(highlight_path))
        self.shift_video(video, start)
        frames = []
        for i in range(100):
            ret, frame = video.read()
            frames.append(frame)

        video.release()
        return np.array(frames), None, vid_path

    @staticmethod
    def shift_video(video, amt):
        video.set(cv2.CAP_PROP_POS_FRAMES, amt)
