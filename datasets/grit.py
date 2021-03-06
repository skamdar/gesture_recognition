import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from scipy.ndimage import filters
import numpy as np

from utils import load_value_file

def cnnlstm(clip, sample_duration):

    h,w = np.array(clip[0]).shape
    #print("fsdf".format(img.size()))

    ret_img = np.zeros((1, h, w))
    ret_clip = []
    #clip = [img.convert('LA') for img in clip]
    #print(sample_duration)
    for i in range(1, sample_duration - 1):
        #print(np.array(clip[i]).shape)
        ret_clip.append(Image.fromarray(np.bitwise_and(np.array(clip[i]) - np.array(clip[i - 1])),
                        (np.array(clip[i + 1]) - np.array(clip[i]))))


    return torch.from_numpy(ret_img).float()



def cal_gradient(clip, sample_duration):

    h,w = np.array(clip[0]).shape
    #print("fsdf".format(img.size()))

    ret_img = np.zeros((3, h, w))

    #clip = [img.convert('LA') for img in clip]
    #print(sample_duration)
    for i in range(1, sample_duration):
        #print(np.array(clip[i]).shape)
        ret_img[0,:,:] = ret_img[0,:,:] + (np.array(clip[i]) - np.array(clip[i - 1]))

    img = Image.fromarray(ret_img[:,:,0])
    filters.sobel(img, 1, ret_img[:, :, 1])
    filters.sobel(img, 0, ret_img[:, :, 2])

    #print(type(ret_img))
    #img = Image.fromarray(ret_img)

    return torch.from_numpy(ret_img).float()


def pil_loader(path):
    #print("pil_loader called")
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            #return img.convert('RGB')
            ## MCCNN
            return img.convert('L')


def accimage_loader(path):
    #print("accimage loader called")
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    pth = video_dir_path.split('/')
    pth = pth[-1].split('_')
    #print(video_dir_path)
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:2d}.jpg'.format(i - 1))
        #print(image_path)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    #print("get default video loader called")
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    # video_names contains:
    # [ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01, ApplyLipstick/v_ApplyLipstick_g17_c05, ...]
    # annotations contains:
    # [{'label':'ApplyMakeup'}, {'label':'ApplyLipstick'}, ...]

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        # n_sample_for_each_video = number of samples to create from a video
        # sample_duration = number of frames in each sample

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        # we are done here, else part is not important as of now
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class GRIT(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 sample_duration,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.sample_duration = sample_duration

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        #print(path)
        #print(frame_indices)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        if self.spatial_transform is not None:
            #print(self.spatial_transform)
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        ######### important
        ######### inject gradient info in channels
        #clip = cal_gradient(clip, self.sample_duration)
        ###MCCNN
        # clip = torch.unsqueeze(clip, 0).permute(1, 0, 2, 3)
        #########
        # torch.stack concatenate vector entries in a new tensor along new(0) dimension
        #print(clip.size)

        ### CNNLSTM
        clip = cnnlstm(clip, self.sample_duration)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        # returns:
        # clip: an array of images
        # target: dictionary containing sample info

        return clip, target

    def __len__(self):
        return len(self.data)
