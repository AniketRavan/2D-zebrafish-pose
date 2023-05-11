import numpy as np
import scipy.io as sio
from avi_r import AVIReader
import pdb

def readVideo(path):
    vidobj = AVIReader(path)
    vid_arr = []
    for frame in vidobj:
        gray_channel = frame.numpy()[:, :, 0]
        vid_arr.append(gray_channel)
    vid_numpy = np.stack(vid_arr, axis = 2)
    return vid_numpy



def bgsub(video):
    nframes = video.shape[2]
    nSampFrame = np.min([np.fix(nframes / 2), 100])
    sampFrame = video[:, :, np.fix(np.linspace(0, nframes - 1, int(nSampFrame))).astype(int)]
    distinctFrames_sorted = np.sort(sampFrame, axis = 2)

    # Find the pixel value that is brighter than 90% of the frames and set that as the background pixel value
    videobg = distinctFrames_sorted[:, :, int(np.fix(nSampFrame * 0.9))]

    # Subtract background from video
    video1 = np.zeros((videobg.shape[0], videobg.shape[1], nframes))
    for n0 in range(0, nframes):
        video1[:, :, n0] = videobg - video[:, :, n0]

    return video1
