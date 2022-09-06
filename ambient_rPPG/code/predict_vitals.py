import tensorflow as tf
import numpy as np
import scipy.io
# import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
import scipy
import matplotlib.pyplot as plt


def predict_vitals(video_file_path: str,
                   sampling_rate: int = 30,
                   batch_size: int = 100,
                   ckpt: str = r'../mtts_can.hdf5',
                   debug=False):    
    """
    Input 
    video_file_path (str) : path of the video
    sampling_rate (int) : ... 뭔지 잘 모르겠습니다. 아마도 fps?
    batch_size (int) : multiplier of 10 (왜 10의 배수여야하는지, 잘 모르겠습니다.)
    ckpt (int) : path of saved pretrained weight

    Return
    pulse_pred (numpy ndarray) : predicted pulse wave
    resp_pred (numpy ndarray) : prediected respiration
    fps (float etc.)
    """

    img_rows = 36
    img_cols = 36
    frame_depth = 10                # ??? what is the frame depth???
    fs = sampling_rate
    sample_data_path = video_file_path
    model_checkpoint = ckpt

    dXsub, fps = preprocess_raw_video(sample_data_path, dim=36, debug=debug)
    if debug:
        print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    # part to delete
    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    return pulse_pred, resp_pred, fps

def predict_hr(pulse_pred, fps):
    max_bpm = 200                       # max BPM = 220 - age. BPM over max_bpm is considered as a noise or an error.
    peak_ignore_frame = int(60 * fps / max_bpm + 1e-5)
    assert 3 < peak_ignore_frame < 10, '죄송합니다. 이 에러가 발생했다면 제가 코드를 잘못 짠 것입니다.'
    peaks, _ = scipy.signal.find_peaks(pulse_pred, distance=peak_ignore_frame)
    # ibi : intertbeat interval
    ibi = np.diff(peaks) / fps
    avg_ibi = np.average(ibi)
    avg_bpm = 60 / avg_ibi
    return avg_bpm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    parser.add_argument('--ckpt', type=str, defalut = '../mtts_can.hdf5', help='checkpoint of model')
    args = parser.parse_args()

    pulse_pred, resp_pred, fps = predict_vitals(args)
    avg_bpm = predict_hr(pulse_pred, fps)

    ########## Plot ##################
    plt.subplot(211)
    plt.plot(pulse_pred)
    plt.title('Pulse Prediction')
    plt.subplot(212)
    plt.plot(resp_pred)
    plt.title('Resp Prediction')
    plt.show()
