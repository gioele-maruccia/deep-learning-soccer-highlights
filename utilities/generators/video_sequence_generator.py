# import the necessary packages
from __future__ import division
import cv2
import numpy as np
from statistics import mean
from pathlib import Path
import os

# generate random integer values
from random import seed
from random import randint


class VideoSequenceGenerator:
    def __init__(self, clip_size):
        self.clip_size = clip_size

    def generate(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        info_list = []
        i = pad_frames
        video = cv2.VideoCapture(vid_path)
        vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(vid_path.split(os.path.sep)[-1])
        parent = Path(vid_path).parent.parent
        out_dir = parent / 'sequences'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_dir = out_dir / ('full_game_goal_cuts_seq_' + str(self.clip_size))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if generate:
            fps = int(video.get(cv2.CAP_PROP_FPS))
            vid_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
            video_name_with_fmt = vid_path.split(os.path.sep)[-1]
            num_seq = 0
            frames = []

            parent_highlight_len = vid_len - 2 * pad_frames

            video.set(cv2.CAP_PROP_POS_FRAMES, pad_frames)
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                height, width, layers = frame.shape

                frames.append(frame)

                if (i != 0 and len(frames) == self.clip_size) and (i <= vid_len - pad_frames):
                    # if (i != 0 and i % self.clip_size == 0) and (i <= vid_len-pad_frames):
                    num_seq += 1
                    out_vid_path = out_dir / ('seq_' + str(num_seq) + '__' + video_name_with_fmt)

                    # check if video is alredy present and is valid for processing
                    if os.path.isfile(out_vid_path):
                        present_vid = cv2.VideoCapture(str(out_vid_path))
                        if int(present_vid.get(cv2.CAP_PROP_FRAME_COUNT)) != self.clip_size:
                            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                            out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
                            for i in range(len(frames)):
                                # writing to a image array
                                out.write(frames[i])
                            out.release()
                    else:
                        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
                        for i in range(len(frames)):
                            # writing to a image array
                            out.write(frames[i])
                        out.release()

                    if (i - self.clip_size + 1) < pad_frames or (i - self.clip_size + 1) >= (vid_len - pad_frames):
                        isHighlight = False
                    else:
                        isHighlight = True
                    if (i - self.clip_size) >= ignore_frame:
                        isCelebration = True
                    else:
                        isCelebration = False

                    goalDistance = i - max_frame

                    start_seq = i - self.clip_size + 1
                    info_list.append(
                        [str(out_vid_path), isHighlight, isCelebration, start_seq, max_frame, parent_highlight_len,
                         pad_frames])

                    frames = []
                i += 1

        video.release()

        if info_list:
            return info_list, out_dir
        else:
            return None, out_dir

    def create_path(self, path):
        parent = Path(path).parent.parent
        out_dir = parent / 'sequences'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_dir = out_dir / ('full_game_goal_cuts_seq_' + str(self.clip_size))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        return out_dir

    @staticmethod
    def get_video_info(video, path):
        fps = int(video.get(cv2.CAP_PROP_FPS))
        vid_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_name_with_fmt = path.split(os.path.sep)[-1]
        return [fps, vid_len, video_name_with_fmt]

    @staticmethod
    def shift_video(video, amt):
        video.set(cv2.CAP_PROP_POS_FRAMES, amt)

    def write_sequence(self, out_vid_path, frames, fps, width, height):
        if os.path.isfile(out_vid_path):
            present_vid = cv2.VideoCapture(str(out_vid_path))
            if int(present_vid.get(cv2.CAP_PROP_FRAME_COUNT)) != self.clip_size:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
                for i in range(len(frames)):
                    # writing to a image array
                    out.write(frames[i])
                out.release()
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))
            for i in range(len(frames)):
                # writing to a image array
                out.write(frames[i])
            out.release()

    def alredy_exists(self, out_vid_path):
        if os.path.isfile(out_vid_path):
            present_vid = cv2.VideoCapture(str(out_vid_path))
            if int(present_vid.get(cv2.CAP_PROP_FRAME_COUNT)) != self.clip_size:
                return False
            else:
                return True
        else:
            return False

    def get_seq_state(self, num_seq, no_high_seqs, i, ignore_frame):
        if num_seq <= no_high_seqs:
            is_highlight = False
        else:
            is_highlight = True
        if (i - self.clip_size) >= ignore_frame:
            is_celebration = True
        else:
            is_celebration = False

        start_seq = i - self.clip_size
        return is_highlight, is_celebration, start_seq


    def generate2_old(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []
        i = 1
        no_high_seqs = 2

        if generate:
            num_seq = 0
            frames = []
            video = cv2.VideoCapture(vid_path)
            [fps, vid_len, video_name_with_fmt] = self.get_video_info(video, vid_path)
            full_highlight_len = vid_len - 2 * pad_frames
            self.shift_video(video, pad_frames - no_high_seqs * self.clip_size)
            ignore_frame = ignore_frame - (pad_frames - no_high_seqs * self.clip_size)
            # 2 sequences added as non-relevant video
            tot_sequences = no_high_seqs + int(full_highlight_len / self.clip_size)
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                height, width, layers = frame.shape
                # we process only a finite number of sequences, that means it could certainly happens that the last
                # odd part of the highlight will be lost (final celebration part [not relevant])
                if num_seq == tot_sequences:
                    continue

                frames.append(frame)
                if len(frames) == self.clip_size:
                    num_seq += 1
                    out_vid_path = out_dir / ('seq_' + str(num_seq) + '__' + video_name_with_fmt)
                    self.write_sequence(out_vid_path, frames, fps, width, height)
                    is_highlight, is_celebration, start_seq = self.get_seq_state(num_seq, no_high_seqs, i, ignore_frame)

                    info_list.append([str(out_vid_path), is_highlight, is_celebration, start_seq, max_frame, full_highlight_len, pad_frames])

                    frames = []
                i += 1
            video.release()
        if info_list:
            return info_list, out_dir

        return None, out_dir


    def get_seq_state_r(self, len, shift, pad_frames, ignore_frame):
        # TODO: chiedere  a Mirto
        if shift > pad_frames - len/5:
            is_highlight = True
        else:
            is_highlight = False

        if shift >= ignore_frame - len/2:
            is_celebration = True
        else:
            is_celebration = False

        return is_highlight, is_celebration



    def generate2(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []
        no_high_seqs = 2
        no_high_seq_len = no_high_seqs * self.clip_size
        useless_frames = pad_frames - no_high_seq_len
        max_frame -= useless_frames
        ignore_frame -= useless_frames
        if generate:
            video = cv2.VideoCapture(vid_path)
            [fps, vid_len, video_name_with_fmt] = self.get_video_info(video, vid_path)
            real_highlight_len = vid_len - 2 * pad_frames
            # 2 sequences added as non-relevant video
            tot_sequences = no_high_seqs + int(real_highlight_len / self.clip_size)

            # generate 2xtot_sequences sequences starting at random position into the video
            for _ in range(4 * tot_sequences):
                shift = randint(useless_frames, pad_frames + real_highlight_len - self.clip_size)
                out_vid_path = out_dir / ('seq_frame' + str(shift) + '__' + video_name_with_fmt)
                num_seq = 0
                frames = []
                self.shift_video(video, shift)
                if not self.alredy_exists(out_vid_path):
                    while len(frames) < self.clip_size:
                        ret, frame = video.read()
                        if not ret:
                            break
                        height, width, layers = frame.shape
                        frames.append(frame)
                        num_seq += 1
                    self.write_sequence(out_vid_path, frames, fps, width, height)
                is_highlight, is_celebration = self.get_seq_state_r(self.clip_size, shift, pad_frames, ignore_frame)
                print(str(out_vid_path))
                info_list.append([str(out_vid_path), is_highlight, is_celebration, shift-useless_frames, max_frame, real_highlight_len, no_high_seq_len, ignore_frame])
                frames = []

            video.release()
        if info_list:
            return info_list, out_dir

        return None, out_dir

    def generate_all(self, vid_path, max_frame, pad_frames, ignore_frame, generate):
        print("*")
        out_dir = self.create_path(vid_path)
        info_list = []
        no_high_seqs = 2
        no_high_seq_len = no_high_seqs * self.clip_size
        useless_frames = pad_frames - no_high_seq_len
        max_frame -= useless_frames
        ignore_frame -= useless_frames
        if generate:
            video = cv2.VideoCapture(vid_path)
            [fps, vid_len, video_name_with_fmt] = self.get_video_info(video, vid_path)
            real_highlight_len = vid_len - 2 * pad_frames
            # 2 sequences added as non-relevant video
            tot_sequences = no_high_seqs + int(real_highlight_len / self.clip_size)

            # generate 2xtot_sequences sequences starting at random position into the video
            # seed random number generator
            for shift in range(useless_frames, pad_frames + int(real_highlight_len) - self.clip_size + 1, 10):
                print(shift)
                out_vid_path = out_dir / ('seq_frame' + str(shift) + '__' + video_name_with_fmt)
                num_seq = 0
                frames = []
                self.shift_video(video, shift)
                if not self.alredy_exists(out_vid_path):
                    while len(frames) < self.clip_size:
                        ret, frame = video.read()
                        if not ret:
                            break
                        height, width, layers = frame.shape
                        frames.append(frame)
                        num_seq += 1
                    self.write_sequence(out_vid_path, frames, fps, width, height)
                is_highlight, is_celebration = self.get_seq_state_r(self.clip_size, shift, pad_frames, ignore_frame)
                info_list.append([str(out_vid_path), is_highlight, is_celebration, shift-useless_frames, max_frame, real_highlight_len, no_high_seq_len, ignore_frame])
                frames = []

            video.release()
        if info_list:
            return info_list, out_dir

        return None, out_dir