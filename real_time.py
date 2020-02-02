#done changes
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from threading import Thread
from threading import Lock
import time
import cv2
from torch.autograd import Variable
import torch.nn.functional as F

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test


def readwebcam():
    cap = cv2.VideoCapture(0)
    global frames

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(frame.shape)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        frame = np.transpose(frame, (2, 0, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        lock.acquire()
        frames = np.delete(frames, (0), axis=0)
        frames = np.append(frames, frame.reshape(1, 3, 480, -1), axis=0)
        lock.release()
        #time.sleep(0.1)
    # When everything done, release the capture
    cap.release()


if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)


    """if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
    """

    lock = Lock()
    t = Thread(target=readwebcam)
    t.start()
    frames = np.random.rand(16, 3, 480, 640)

    model.eval()
    classes = ['abort', 'circle', 'hello', 'no', 'stop', 'turn', 'turn_left', 'turn_right', 'warn']

    while (True):
        lock.acquire()
        print(frames[1])
        lock.release()
        # inputs type and shape
        #<class 'torch.Tensor'>
        #torch.Size([10, 3, 16, 112, 112])
        #TODO add transformations
        inputs = torch.unsqueeze(torch.from_numpy(frames), 0).permute(0, 2, 1, 3, 4)
        print(inputs.size())
        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
        outputs = F.softmax(outputs)
        _, ind = torch.max(outputs)
        print(classes[ind])
        #time.sleep(1)

    t.join()