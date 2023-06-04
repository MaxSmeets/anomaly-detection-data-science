import os
os.environ['LD_LIBRARY_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib'

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import csv


# Dictionary to map joints of body part
KEYPOINT_DICT = {
    'nose':0,
    'left_eye':1,
    'right_eye':2,
    'left_ear':3,
    'right_ear':4,
    'left_shoulder':5,
    'right_shoulder':6,
    'left_elbow':7,
    'right_elbow':8,
    'left_wrist':9,
    'right_wrist':10,
    'left_hip':11,
    'right_hip':12,
    'left_knee':13,
    'right_knee':14,
    'left_ankle':15,
    'right_ankle':16
} 

def run_pose_estimation(video_path):

    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    
    # Print the keys of model signatures
    print(model.signatures.keys())

    movenet = model.signatures['serving_default']

    video = cv2.VideoCapture(video_path)
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    csv_filename = 'pose_estimation_data.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Keypoint', 'X', 'Y', 'Score'])

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (12, 12))
        input_data = tf.cast(tf.expand_dims(frame_resized, axis=0), dtype=tf.int32)

        outputs = movenet(input_data)

        # Retrieve keypoints
        keypoints_key = [key for key in outputs.keys() if 'output' in key]
        if not keypoints_key:
            raise ValueError("Pose estimation data not found in the outputs dictionary.")

        keypoints = outputs[keypoints_key[0]].numpy()[0]

        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            x, y, score = keypoint[1], keypoint[0], keypoint[2]
            csv_writer.writerow([frame_count, i, x, y, score])

        frame_count += 1

    csv_file.close()
    video.release()

    print(f"Pose estimation completed. Results saved to {csv_filename}")


video_path = r'D:\HBO_ICT_ZUYD\BD04_Data_Science\Videos_movenet\test.mp4'
run_pose_estimation(video_path)
