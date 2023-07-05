import math
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# global configs
CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE, size2 = 16, 128, 108, 200 


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'modal1'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
    """

    def __init__(self, dataset='modal1', split='train'):
        self.original_dir = os.path.join('/DATA/rani.1/feb', dataset) 
        self.preprocessed_dir = os.path.join('/DATA/rani.1/feb/',dataset, 'preprocessed_' + dataset)
        self.split = split

        if not self.check_integrity():
            raise RuntimeError('{} split of {} dataset is not found. You need to '
                               'download it from official website.'.format(split, dataset))

        if not self.check_preprocess():
            print('Preprocessing {} split of {} dataset, this will take long, '
                  'but it will be done only once.'.format(split, dataset))
            self.preprocess()

        self.file_names, labels = [], []
        for label in sorted(os.listdir(os.path.join(self.preprocessed_dir, self.split))):
            for file_name in sorted(os.listdir(os.path.join(self.preprocessed_dir, self.split, label))):
                self.file_names.append(os.path.join(self.preprocessed_dir, self.split, label, file_name))
                labels.append(label)

        print('Number of {} videos: {:d}'.format(split, len(self.file_names)))

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(get_labels(dataset))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # load and preprocess.
        #Intialize
        buffer = self.load_frames(self.file_names[index])
        label = np.array(self.label_array[index])
        if self.split == 'train':
            try:
                buffer1 = self.crop(buffer,CLIP_LEN,CROP_SIZE,size2)
            except ValueError:
                buffer1 = self.crop(buffer,CLIP_LEN,CROP_SIZE,152)
        else:
            if self.split == 'val':
                buffer1 = self.crop(buffer,CLIP_LEN,CROP_SIZE,size2)
            else:
                buffer1 = self.check(buffer,CLIP_LEN)

        buffer1 = self.resize(buffer1)
        buffer1 = self.normalize(buffer1)
        buffer1 = self.to_tensor(buffer1)
        return torch.from_numpy(buffer1), torch.from_numpy(label), self.file_names[index]

    def check_integrity(self):
        if os.path.exists(os.path.join(self.original_dir, self.split)):
            return True
        else:
            return False

    def check_preprocess(self):
        if os.path.exists(os.path.join(self.preprocessed_dir, self.split)):
            return True
        else:
            return False

    def preprocess(self):
        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)
        os.mkdir(os.path.join(self.preprocessed_dir, self.split))

        for file in sorted(os.listdir(os.path.join(self.original_dir, self.split))):
            os.mkdir(os.path.join(self.preprocessed_dir, self.split, file))

            for video in sorted(os.listdir(os.path.join(self.original_dir, self.split, file))):
                video_name = os.path.join(self.original_dir, self.split, file, video)
                save_name = os.path.join(self.preprocessed_dir, self.split, file, video)
                self.process_video(video_name, save_name)

        print('Preprocess finished.')

    @staticmethod
    def process_video(video_name, save_name):
        print('Preprocess {}'.format(video_name))
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(video_name)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # make sure the preprocessed video has at least CLIP_LEN frames
        extract_frequency = 4
        if frame_count // extract_frequency <= CLIP_LEN:
            extract_frequency -= 1
            if frame_count // extract_frequency <= CLIP_LEN:
                extract_frequency -= 1
                if frame_count // extract_frequency <= CLIP_LEN:
                    extract_frequency -= 1

        count, i, retaining = 0, 0, True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % extract_frequency == 0:
                resize_height = RESIZE_HEIGHT
                resize_width = math.floor(frame_width / frame_height * resize_height)
                # make sure resize width >= crop size
                if resize_width < CROP_SIZE:
                    resize_width = RESIZE_HEIGHT
                    resize_height = math.floor(frame_height / frame_width * resize_width)

                frame = cv2.resize(frame, (resize_width, resize_height))
                s = str(save_name.split('.')[0]+"."+str(save_name.split('.')[1]))
                if not os.path.exists(s):
                    os.mkdir(s)
                cv2.imwrite(filename=os.path.join(s, '{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

    @staticmethod
    def resize(buffer):
        n=buffer.shape[0]
        new_buffer=np.empty((n,112,128,3))
        for i, frame in enumerate(buffer):
            frame=cv2.resize(buffer[i], (128,112), interpolation= cv2.INTER_LINEAR)
            new_buffer[i]=frame
        return new_buffer

    @staticmethod
    def normalize(buffer):
        buffer = buffer.astype(np.float32)
        for i, frame in enumerate(buffer):
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    @staticmethod
    def to_tensor(buffer):
        return buffer.transpose((3, 0, 1, 2))

    @staticmethod
    def load_frames(file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = []
        total = 0
        for i, frame_name in enumerate(frames):
            total = len(frames)
            skip = int(total/16) #uniform frame loading for HGD dataset
            try:
                if not i% skip:
                    frame = np.array(cv2.imread(frame_name))
                    buffer.append(frame)
                else:
                    pass
            except ZeroDivisionError:
                skip= int(total/10) #uniform frame loading for Autsim + HMDB51 dataset
                if total >=10:
                    if not i%skip:
                        frame = np.array(cv2.imread(frame_name))
                        buffer.append(frame)
                else:
                    frame = np.array(cv2.imread(frame_name))
                    buffer.append(frame)
        return np.array(buffer).astype(np.uint8)
    

    def crop(self, buffer, clip_len, crop_size,crop_size2):
        if self.split == 'train':
            # randomly select time index for temporal jitter
            if buffer.shape[0] > clip_len:
                time_index = random.randint(0,buffer.shape[0] - clip_len)
            else:
                time_index = 0
            # randomly select start indices in order to crop the video
            height_index = random.randint(0,buffer.shape[1] - crop_size)
            width_index = random.randint(0,buffer.shape[2] - crop_size2)
            # crop and jitter the video using indexing. The spatial crop is performed on
            # the entire array, so each frame is cropped in the same location. The temporal
        else:
            # for val and test, select the middle and center frames
            if buffer.shape[0] > clip_len:
                #time_index=0
                time_index = math.floor((buffer.shape[0] - clip_len) / 2)
            else:
                time_index = 0
            height_index = math.floor((buffer.shape[1] - crop_size) / 2)
            width_index = math.floor((buffer.shape[2] - crop_size2) / 2)
        buffer = buffer[time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size2, :]

        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer
        
        
    def check(self,buffer,clip_len):
        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        else:
            time_index=0
            buffer = buffer[time_index:time_index + clip_len :]
        return buffer    

def get_labels(dataset='ucf101'):
    labels = []
    with open('/DATA/rani.1/feb/{}/{}_labels.txt'.format(dataset,dataset), 'r') as load_f:
        raw_labels = load_f.readlines()
    for label in raw_labels:
        labels.append(label.replace('\n', ''))
    return sorted(labels)

