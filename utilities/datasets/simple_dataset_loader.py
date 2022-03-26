import numpy as np
import cv2
import os
import pandas as pd
import cv2
import progressbar
from utilities.generators import VideoSequenceGenerator
from pathlib import Path

from itertools import islice
from utilities.preprocessing import VideoVGG16FeatureExtractor
from utilities.preprocessing import VideoScorerPreprocessor


# import psutil


class SimpleDatasetLoader:
    """ class able to load specific image or video data and apply to them dynamic generators and preprocessors.
    While a PREPROCESSOR in a one input-one output function able to transform in some way the input image/video, a GENERATOR take one
    file to generate many outputs for it (i.e. the simple splitting of a video in several subsequences"""
    frame_count = 0

    def __init__(self, preprocessors=None, generators=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        self.generators = generators

        # if the preprocessors or generators are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []
        # note: if no generator has been provided, the null generator should be added by default
        if self.generators is None:
            self.generators = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return np.array(data), np.array(labels)

    # def video_transform(self, video_list):
    #     """ makes an input-output transformation of data taken from a list """
    #     for p in self.preprocessors:
    #         [video, label, path] = p.preprocess(str(path), isHighlight, isCelebration, int(initSeqFrame), int(maxFrame),
    #                                             float(highlight_length), int(pad_frames), int(start_celebration))


    def load_video(self, video_csv_path, verbose=-1, bs=None, num_images=0, mode="train"):
        sequences2load = 0

        f = open(video_csv_path, "r")
        # read header
        line = f.readline()

        while True:
            # gives an object with many fields
            # print(psutil.virtual_memory())

            # initialize the list of features and labels
            data = []
            labels = []
            while len(data) < bs:
                video = []
                line = f.readline()
                if line == "":
                    print("ciao")
                    # reset the file pointer to the beginning of the file
                    # and re-read the line
                    f.seek(0)
                    line = f.readline()
                    line = f.readline()
                    # if we are evaluating we should now break from our
                    # loop to ensure we don't continue to fill up the
                    # batch from samples at the beginning of the file
                    if mode == "eval":
                        break
                # extract the label and construct the image
                line = line.strip().split(",")
                path = line[0]
                isHighlight = line[1]
                isCelebration = line[2]
                initSeqFrame = line[3]
                maxFrame = line[4]
                highlight_length = line[5]
                pad_frames = float(line[6])
                start_celebration = float(line[7])
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        # if isinstance(p, VideoVGG16FeatureExtractor):
                        #     [video, label, path] = p.preprocess_mod(str(path), video, isHighlight, isCelebration, int(initSeqFrame), int(maxFrame), float(highlight_length), int(pad_frames), int(start_celebration))
                        # elif isinstance(p, VideoScorerPreprocessor):
                        #     [video, label, path] = p.preprocess(str(path), video, isHighlight, isCelebration, int(initSeqFrame), int(maxFrame), float(highlight_length), int(pad_frames), int(start_celebration))
                        # else:
                        #     [video, label, path] = p.preprocess_and_save(str(path), video, isHighlight, isCelebration,
                        #                                     int(initSeqFrame), int(maxFrame), float(highlight_length),
                        #                                     int(pad_frames), int(start_celebration))
                        [video, label, path] = p.preprocess_and_save(str(path), video, isHighlight, isCelebration,
                                                        int(initSeqFrame), int(maxFrame), float(highlight_length),
                                                        int(pad_frames), int(start_celebration))

                        #[video, label, path] = p.preprocess(str(path), video, isHighlight, isCelebration, int(initSeqFrame), int(maxFrame), float(highlight_length), int(pad_frames), int(start_celebration))

                    # by updating the data list followed by the labels
                    if labels is not None:
                        data.append(video)
                        label = np.array(label)
                        # label = np.transpose(label[np.newaxis])
                        labels.append(label)
                        # show an update every 'verbose' images
                if verbose > 0 and len(data) > 0 and (len(data) + 1) % verbose == 0:
                    print("\n[INFO] loaded video {}/{}".format(len(data) + 1, sequences2load))

            #labels = np.array(labels)
            #print("labels.shape = ", labels.shape)

            #yield np.array(data).astype("float") / 255.0, labels <--- vecchia maniera
            #yield np.array(data), labels
            yield np.squeeze(np.array(data)), np.squeeze(labels)

            # return a tuple of the data and labels
            # return np.array(data), np.array(labels)

    def load_dataset(self, video_csv_path, generate=False, try_one_video=False):
        for x in self.generators:
            if isinstance(x, VideoSequenceGenerator):
                print("[INFO] creating sequences...")
                df = pd.read_csv(video_csv_path)
                info_list = []
                for index, row in df.iloc[0:].iterrows():
                    video_path = row['video_name']
                    starting_frame = row['starting_frame']
                    added_frames_bf = row['added_frames_bf']
                    goal_frame = row['goal_frame'] - starting_frame + added_frames_bf
                    start_celebration = row['start_celebration'] - starting_frame + added_frames_bf
                    if try_one_video:
                        tmp_list, dir_path = x.generate_all(str(video_path), goal_frame, added_frames_bf, start_celebration, generate)
                    else:
                        tmp_list, dir_path = x.generate2(str(video_path), goal_frame, added_frames_bf, start_celebration, generate)
                    if tmp_list is not None:
                        info_list.extend(tmp_list)

                if info_list:
                    if not try_one_video:
                        np.random.seed(42) # to make randomization reproducible
                        np.random.shuffle(info_list)
                    out_df = pd.DataFrame(info_list,
                                          columns=['out_vid_path', 'isHighlight', 'isCelebration', 'init_seq_frame', 'maxFrame',
                                                   'parent_length', 'pad_frames', 'startCelebration'])

                else:
                    out_df = None

                csv_name = video_csv_path.split(os.path.sep)[-1]

                csv_to_read = dir_path / ('seq_' + csv_name)
                if out_df is not None:
                    if os.path.exists(csv_to_read):
                        os.remove(csv_to_read)
                    out_df.to_csv(csv_to_read, index=False)

        return csv_to_read

    def video2pic(self, cuts_dir, verbose=-1):
        ''' extract frames of celebration and no-celebration
        in order to train the event right part of the highlight window'''
        parent_dir = os.path.dirname(os.path.dirname(cuts_dir))
        celeb_dataset_dir = os.path.join(os.path.join(parent_dir, 'datasets'), 'celeb_dataset')
        if not os.path.exists(celeb_dataset_dir):
            os.makedirs(celeb_dataset_dir)
        celeb_dir = os.path.join(celeb_dataset_dir, 'celebration')
        if not os.path.exists(celeb_dir):
            os.makedirs(celeb_dir)
        no_celeb_dir = os.path.join(celeb_dataset_dir, 'no_celebration')
        if not os.path.exists(no_celeb_dir):
            os.makedirs(no_celeb_dir)

        # get the csv list of all highlight cuts
        label_list = os.path.join(cuts_dir, 'labels_info.csv')
        cuts = pd.read_csv(label_list, sep=',', header='infer')

        # iterate over video cuts and extract celebrations.
        # create folder with celebration / no-celebration frames
        for row in progressbar.progressbar(cuts.itertuples(index=True, name='Pandas')):
            cut_path = getattr(row, "video_name")
            goal_frame = getattr(row, "starting_frame") - getattr(row, "starting_frame") + getattr(row,
                                                                                                   "added_frames_bf")
            start_frame = getattr(row, "added_frames_bf")
            start_celebration_frame = getattr(row, "start_celebration") - getattr(row, "starting_frame") + getattr(row,
                                                                                                                   "added_frames_bf")
            end_celebration_frame = getattr(row, "ending_frame") - getattr(row, "starting_frame") + getattr(row,
                                                                                                            "added_frames_bf")
            cut_name = cut_path.split(os.path.sep)[-1]
            cut_path = os.path.join(cuts_dir, cut_name)

            # extract non celebration frames and put them into a dataset folder
            SimpleDatasetLoader.frame_capture(cut_path, no_celeb_dir, start_frame, start_celebration_frame - 1)
            # extract celebration frames
            SimpleDatasetLoader.frame_capture(cut_path, celeb_dir, start_celebration_frame,
                                              end_celebration_frame)

    # Function to extract frames
    @staticmethod
    def frame_capture(input_path, output_path, start, stop):
        vidObj = cv2.VideoCapture(input_path)
        # Used as counter variable
        count = 0
        # checks whether frames were extracted
        success = 1
        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()
            if start <= count <= stop:
                # Saves the frames with frame-count
                to_write = os.path.join(output_path, "frame%d.jpg" % SimpleDatasetLoader.frame_count)
                print(to_write)
                cv2.imwrite(to_write, image)
                SimpleDatasetLoader.frame_count += 1
            count += 1