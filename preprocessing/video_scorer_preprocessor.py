# import the necessary packages
from __future__ import division
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class VideoScorerPreprocessor:
    def __init__(self, clip_size, verbose=10):
        self.verbose = verbose
        self.clip_size = clip_size

    def preprocess_and_save(self, video_path, video, isHighlight, isCelebration, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration):
        X = np.load(video_path, mmap_mode='r')
        #X = np.array(video)
        len = X.shape[0]
        # labels = self.polynomial(len, initSeqFrame, maxFrame, highlight_length, pad_frames)
        labels = self.gaussian(len, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration)
        return X, labels, video_path

    def preprocess(self, video_path, video, isHighlight, isCelebration, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration):
        #X = np.load(video_path, mmap_mode='r')
        X = np.array(video)
        len = X.shape[0]
        # labels = self.polynomial(len, initSeqFrame, maxFrame, highlight_length, pad_frames)
        labels = self.gaussian(len, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration)
        return video, labels, video_path

    def polynomial(self, len, initSeqFrame, maxFrame, highlight_length, pad_frames):
        x = np.array([pad_frames, maxFrame, pad_frames + int(highlight_length)])
        y = np.array([0, 1, 0])

        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)
        all_xp = np.linspace(pad_frames, pad_frames + int(highlight_length))
        # _ = plt.plot(x, y, '.', xp, p(xp), '-')

        xp = np.linspace(initSeqFrame, initSeqFrame + len, num=self.clip_size)
        _ = plt.plot(x, y, '.', xp, p(xp), '-', all_xp, p(all_xp), '--')
        plot = False
        if plot:
            plt.ylim(0, 1)
            plt.show()
        return p(xp)

    def gaussian(self, len, initSeqFrame, maxFrame, highlight_length, no_high_seq_len, start_celebration):
        all_x = np.arange(0, no_high_seq_len + int(highlight_length) + self.clip_size, 1)

        # calculate gaussian
        all_y = VideoScorerPreprocessor.gaussian_fun(all_x, maxFrame, (maxFrame - no_high_seq_len) / 2)

        # not isHighlight:
        # all_y[0:no_high_seq_len] = 0

        # isHiglight and isCelebration
        # all_y[start_celebration::] = 0

        x = all_x[initSeqFrame:initSeqFrame + len]
        y = all_y[initSeqFrame:initSeqFrame + len]

        plot = False
        if plot:
            fig = plt.figure()
            _ = plt.plot(x, y, 'o', all_x, all_y, 'r')
            plt.ylim(0, 1)
            plt.draw()
            plt.waitforbuttonpress(0)  # this will wait for indefinite time
            plt.close(fig)

        return y

    @staticmethod
    def gaussian_fun(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def preprocess_2_video(self, video_path, isHighlight, isCelebration, goalDistance):

        cap = cv2.VideoCapture(video_path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint4'))
        fc = 0
        ret = True

        while fc < frameCount and ret:
            ret, buf[fc] = cap.read()
            fc += 1

        cap.release()
        if isHighlight:
            return buf, goalDistance, video_path
        else:
            return None, None, video_path
