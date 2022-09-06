import numpy as np
import cv2
from skimage.util import img_as_float
import matplotlib.pyplot as plt
# import scipy.io
from scipy.sparse import spdiags

def preprocess_raw_video(videoFilePath: str, face_region: tuple = None, dim: int = 36, debug=False):

    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath);
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))         # get the number of all frames
    fps = vidObj.get(cv2.CAP_PROP_FPS)                               # get fps
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)

    if face_region:
        # face_region = (x1, y1, x2, y2) : bounding box coordinate
        left, top, right, bottom = face_region

    #########################################################################
    # Crop each frame size into dim x dim
    success, img = vidObj.read()
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))                 # current timestamp in milisecond
        if face_region:
            vidLxL = cv2.resize(img_as_float(img[top:bottom, left:right, :]), (dim, dim), interpolation = cv2.INTER_AREA)
        else:
            vidLxL = cv2.resize(img_as_float(img), (dim, dim), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE)        # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1

    if debug:
        height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
        print("Orignal Height", height)
        print("Original width", width)
        plt.imshow(Xsub[0])
        plt.title('Sample Preprocessed(cropped) Frame')
        plt.show()

    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)

    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:totalFrames-1, :, :, :]

    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3);

    return dXsub, fps

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal
