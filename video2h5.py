import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import h5py
import skvideo.io
#Function for Feature Extraction

def frame_crop_center(video,cropf):
    f,_,_,_ = video.shape
    startf = f//2 - cropf//2
    return video[startf:startf+cropf, :, :, :]



def extract_tarian(path, frame_size, seq_len):
    list_video = []
    list_label = []
    label_index = 0
    video_dims = []
    for folder in path:
        for f in tqdm(os.listdir(folder)):
            f = os.path.join(folder, f)
        # checking if it is a file
            
            video = skvideo.io.vread(f)
            video_dims.append(video.shape)
            L=[]

            #resize video dimensions
            for i in range(video.shape[0]):
                frame = cv2.resize(video[i], (frame_size,frame_size), interpolation=cv2.INTER_CUBIC)
                L.append(frame)

            video = np.asarray(L)

            #center crop video to have consistent video frame number
            video = frame_crop_center(video, seq_len)

            list_video.append(video)
            list_label.append(label_index)
            
            del video
            gc.collect()
        label_index += 1

        
    return list_video, list_label, video_dims
# label_data = pd.read_csv("/media/virgantara/DATA1/Penelitian/Datasets/HumanMotionDB/hmdb51_org", sep=' ', header=None)


path=[]
dir_path = "/media/virgantara/DATA1/Penelitian/Datasets/HumanMotionDB/hmdb51_org/half1/"
for dir in os.listdir(dir_path):
    path.append(os.path.join(dir_path,dir))

print(path[0])
#

FRAME_SIZE = 224
SEQ_LEN = 25
list_video, list_label, video_dims = extract_tarian(path, frame_size=FRAME_SIZE, seq_len=SEQ_LEN)

videos = []
labels = []
for video,label in zip(list_video,list_label):
    if video.shape[0] == SEQ_LEN:
        videos.append(video)
        labels.append(label)
#         print(np.array(video).shape)

print(np.array(videos).shape)
print(np.array(labels).shape)
videos = np.asarray(videos)
labels = np.asarray(labels)

del list_video
del list_label

import h5py

with h5py.File("dataset_ucf50_"+str(FRAME_SIZE)+"_"+str(SEQ_LEN)+".h5", "w") as f:
    f.create_dataset("videos", data=np.asarray(videos))
    f.create_dataset("labels", data=np.asarray(labels))

