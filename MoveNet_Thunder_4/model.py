import os
os.environ['LD_LIBRARY_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib'

import tensorflow as tf
import tensorflow_hub as hub
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
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (256, 256))
        input_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) / 255.0

        input_frame = tf.cast(input_frame, tf.int32)  # Convert input frame to int32
        input_frame = tf.expand_dims(input_frame, axis=0)

        # Run model inference.
        outputs = movenet(input_frame)

        # Print the keys of the outputs dictionary
        print(outputs.keys())

        # Print the complete content of the outputs dictionary
        print(outputs)

        # Verify the correct key for pose estimation data
        if 'output_0' in outputs:
            keypoints = outputs['output_0']
        else:
            raise ValueError("Pose estimation data not found in the outputs dictionary.")

        # Print the shape of the value corresponding to 'output_0'
        print(outputs['output_0'].shape)

        # Check the keys present in the outputs dictionary
        output_keys = list(outputs.keys())
        keypoints_key = [key for key in output_keys if 'output_0' in key]
        scores_key = [key for key in output_keys if 'output_1' in key]

        if not keypoints_key or not scores_key:
            raise ValueError("Pose estimation data not found in the outputs dictionary.")

        keypoints = outputs[keypoints_key[0]].numpy()[0]
        scores = outputs[scores_key[0]].numpy()[0]

        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            score = scores[i]
            x, y = keypoint[1], keypoint[0]
            csv_writer.writerow([frame_count, i, x, y, score])

        frame_count += 1

    csv_file.close()
    video.release()

    print(f"Pose estimation data saved to {csv_filename}")

# Example usage
video_path = r'D:\HBO ICT ZUYD\BD04_Data_Science\Videos_movenet\sessie2023-opdr9-11-zijaanzicht.mp4'
run_pose_estimation(video_path)
