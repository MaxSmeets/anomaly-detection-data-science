import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import csv
import os

path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite')

# Load the MoveNet model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the video
video_path = os.path.join(path, 'test.mp4')
cap = cv2.VideoCapture(video_path)

# Create a CSV file to store the data
csv_file_path = os.path.join(path, 'pose_data.csv')
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Keypoint Score', 'X', 'Y'])

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (256, 256))
    input_data = resized_frame.reshape((1, 256, 256, 3))
    input_data = input_data.astype('float32')
    input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]

    # Convert input data back to UINT8
    input_data = (input_data * 127.5) + 127.5
    input_data = input_data.astype('uint8')

    # Run the inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output
    for detection in output_data:
        keypoints = detection[:, :2]
        confidence_scores = detection[:, 2]

        # Draw keypoints on the frame
        for i, (x, y) in enumerate(keypoints):
            if confidence_scores[i].min() > 0.3:
                cv2.circle(frame, (int(x[0]), int(y[0])), 5, (0, 255, 0), -1)

                # Write data to the CSV file
                csv_writer.writerow([frame_count, confidence_scores[i].min(), x[0], y[0]])

    # Display the frame
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
csv_file.close()