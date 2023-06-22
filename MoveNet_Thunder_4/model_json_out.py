import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os

def process_video_frames(video_path, movenet):
    keypoints_count_list = []  # List to store the counted keypoints
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Resize the frame to the input size required by the MoveNet model
        input_frame = cv2.resize(frame, (256, 256))
        input_frame = tf.expand_dims(input_frame, axis=0)
        input_frame = input_frame / 255.0  # Normalize the input

        # Run the MoveNet model on the frame
        outputs = movenet(input_frame)
        keypoints_count = outputs['output_0'].shape[2]  # Count the number of keypoints

        # Append the count to the list
        keypoints_count_list.append(keypoints_count)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Check if the user pressed 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video.release()
    cv2.destroyAllWindows()
    return keypoints_count_list

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))+ '\\test.mp4'
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    counts = process_video_frames(path, movenet)
    print(counts)

    