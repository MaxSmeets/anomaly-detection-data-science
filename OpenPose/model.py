import cv2
import numpy as np
import csv
from openpose import pyopenpose as op

def run_pose_estimation(video_path):
    # Set up OpenPose parameters
    params = {
        "model_folder": "path_to_openpose_models_folder",
        "net_resolution": "256x256",
        "number_people_max": 1
    }

    # Create OpenPose object
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

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

        # Run pose estimation on the frame
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Extract pose estimation data
        keypoints = datum.poseKeypoints[0]
        scores = np.mean(keypoints[:, :, 2], axis=1)

        # Write the pose estimation data to the CSV file
        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            score = scores[i]
            x, y = keypoint[0], keypoint[1]
            csv_writer.writerow([frame_count, i, x, y, score])

        frame_count += 1

    # Close the CSV file and release the video capture
    csv_file.close()
    video.release()

    print(f"Pose estimation data saved to {csv_filename}")


# Example usage
video_path = 'path_to_your_video.mp4'
run_pose_estimation(video_path)