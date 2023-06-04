import tensorflow as tf
import tensorflow_hub as hub
import cv2
import csv
import numpy as np

def run_pose_estimation(video_path):
    # Load the MoveNet model
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    csv_filename = 'pose_estimation_data.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Keypoint', 'X', 'Y', 'Score'])

    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            # End of video
            break

        # Preprocess the frame
        input_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_data = tf.expand_dims(input_data, axis=0)
        input_data = tf.cast(input_data, dtype=tf.int32)

        # Run pose estimation on the frame
        outputs = movenet(input_data)

        # Verify the correct key for pose estimation data
        if 'output_0' in outputs:
            keypoints = outputs['output_0']
        else:
            raise ValueError("Pose estimation data not found in the outputs dictionary.")

        # Process the outputs
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

        # Display the frame with pose estimation
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the CSV file
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"Pose estimation data saved to {csv_filename}")

if __name__ == '__main__':
    video_path = "D:\HBO ICT ZUYD\BD04_Data_Science\Videos_movenet\test.mp4"
    run_pose_estimation(video_path)
