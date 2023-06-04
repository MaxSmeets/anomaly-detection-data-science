# import os
# os.environ['LD_LIBRARY_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import csv

def run_pose_estimation(video_path):
    # Load MoveNet Thunder 4 model from TensorFlow Hub
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")


    # Open the video file
    video = cv2.VideoCapture(video_path)
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a CSV file to store pose estimation data
    csv_filename = 'pose_estimation_data.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Keypoint', 'X', 'Y', 'Score'])

    frame_count = 0
    while video.isOpened():
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break

        # Resize the frame to match the expected input size for MoveNet
        resized_frame = cv2.resize(frame, (256, 256))

        # Convert the frame to RGB and normalize pixel values
        input_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) / 255.0

        # Add a batch dimension to the input frame
        input_frame = tf.expand_dims(input_frame, axis=0)

        # Run the pose estimation on the frame
        outputs = module(input_frame)

        # Extract pose estimation data from the outputs
        keypoints = outputs['output_0'].numpy()[0]
        scores = outputs['output_1'].numpy()[0]

        # Write the pose estimation data to the CSV file
        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            score = scores[i]
            x, y = keypoint[1], keypoint[0]
            csv_writer.writerow([frame_count, i, x, y, score])

        frame_count += 1

    # Close the CSV file and release the video capture
    csv_file.close()
    video.release()

    print(f"Pose estimation data saved to {csv_filename}")


# Example usage
video_path = r'../Videos_movenet/sessie2023-opdr9-11-zijaanzicht.mp4'
run_pose_estimation(video_path)