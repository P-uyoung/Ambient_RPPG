import os
from glob import glob
import csv
from predict_vitals import predict_vitals
from predict_vitals import predict_hr
import matplotlib.pyplot as plt


def read_label_VIPL_HR_V2(video_folder_path):
    subject_nos = os.listdir(video_folder_path)
    subject_nos = list(map(int, subject_nos))
    subject_nos

    hr = {}
    fps = {}

    for subject_no in subject_nos:
        csv_file_path = video_folder_path + f'/{subject_no}/gt.csv'

        try:
            with open(csv_file_path, 'r') as csv_file:
                rdr = csv.reader(csv_file)
                video_filenames = rdr.__next__()[1:]
                video_hr = rdr.__next__()[1:]
                video_hr = list(map(float, video_hr))
                video_fps = rdr.__next__()[1:]
                video_fps = list(map(float, video_fps))
        except FileNotFoundError:
            continue

        assert len(video_filenames) == len(video_hr) == len(video_fps)

        for i in range(len(video_filenames)):
            hr[f'{subject_no}/{video_filenames[i].strip()}'] = video_hr[i]
            fps[f'{subject_no}/{video_filenames[i].strip()}'] = video_fps[i]

    return hr, fps


def validate_VIPL_HR_V2(video_folder_path, test_num=None):
    """
    Input
    video_folder_path (str) : (absolute) path of the video
    test_num (int) : how many video validate
    """
    errors = []

    cnt = 0
    hr, fps = read_label_VIPL_HR_V2(video_folder_path)
    for video_filename in hr.keys():
        video_file_abspath = os.path.join(video_folder_path, video_filename + '.mp4.avi')
        assert os.path.exists(video_file_abspath), 'File Not Found'
        print(f'inference start : {video_file_abspath}')
        pulse_pred, resp_pred, fps = predict_vitals(video_file_abspath)
        avg_bpm_predict = predict_hr(pulse_pred, fps)
        avg_bpm_gt = hr[video_filename]
        error = avg_bpm_gt - avg_bpm_predict
        errors.append(error)
        print(f'inference done : {video_filename}')
        if test_num and (cnt >= test_num):
            break
        cnt += 1
    
    plt.hist(errors)
    plt.show
