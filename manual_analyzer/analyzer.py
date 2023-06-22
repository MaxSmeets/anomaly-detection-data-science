import cv2
import os
import imageio
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def save_frames(video_path, output_path, start_frame, end_frame, frame_interval):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust start and end frames if they exceed the video length
    start_frame = min(start_frame, frame_count - 1)
    end_frame = min(end_frame, frame_count - 1)

    # Set the current frame to the start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Iterate through frames and save screenshots
    current_frame = start_frame
    while current_frame <= end_frame:
        # Read the current frame
        ret, frame = video.read()

        # Save the frame as a screenshot
        if ret:
            if (current_frame - start_frame) % frame_interval == 0:
                screenshot_path = f"{output_path}/frame_{current_frame}.jpg"
                imageio.imwrite(screenshot_path, frame[:, :, ::-1])
                print(f"Saved screenshot: {screenshot_path}")

        # Move to the next frame
        current_frame += 1

    # Release the video file
    video.release()

def detect_keypoints(image_path, model, movenet):
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    outputs = movenet(image)

    keypoints = outputs['output_0']

    # Convert the TensorFlow tensors to NumPy arrays.
    image_np = image[0].numpy()
    keypoints_np = keypoints[0].numpy()

    # Draw keypoints on the image using OpenCV.
    image_with_keypoints = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for keypoint in keypoints_np:
        x, y, _ = keypoint
        cv2.circle(image_with_keypoints, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

    # Save the image with keypoints.
    output_path = os.path.join(current_dir, 'manual_analyzer\\processed_output')
    cv2.imwrite(output_path, image_with_keypoints)
    print(f"Saved screenshot with keypoints: {output_path}")


# Example usage
current_dir = os.getcwd()
video_path = os.path.join(current_dir, 'manual_analyzer\\test.mp4')
output_path = os.path.join(current_dir, 'manual_analyzer\\output')
start_frame = 400
end_frame = 500
frame_interval = 10

model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# save_frames(video_path, output_path, start_frame, end_frame, frame_interval)
images = os.listdir(output_path)

for image in range(len(images)):
    image_path = os.path.join(current_dir, 'manual_analyzer\\output\\', images[image])
    detect_keypoints(image_path, model, movenet)
