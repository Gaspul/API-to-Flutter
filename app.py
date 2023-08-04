from flask import Flask, request, jsonify
import numpy as np
import base64
import io
from sqprocess import squat
from dlprocess import deadlift
from bpprocess import benchpress

app = Flask(__name__)

@app.route('/process_squat', methods=['POST'])
def process_squat():
    data = request.get_json()  # Retrieve JSON data from the request body
    frame_base64 = data.get('frame')  # Get the base64-encoded video frame from JSON

    # Convert base64 video data to numpy array
    frame_data = base64.b64decode(frame_base64)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

    # Save the received video as an MP4 file
    with open('received_video.mp4', 'wb') as f:
        f.write(frame_np)

    # Perform your pose correction or other processing on 'received_video.mp4'
    # Use the function from mediapipepose.py to process the video
    squat('received_video.mp4', 'processed_video.mp4')

    # Convert the processed video back to base64 string
    with open('processed_video.mp4', 'rb') as f:
        processed_video_data = f.read()

    processed_video_base64 = base64.b64encode(processed_video_data).decode('utf-8')

    # Return the processed video ('processed_video.mp4') as part of the JSON response
    return jsonify({'processed_video': processed_video_base64})


@app.route('/process_deadlift', methods=['POST'])
def process_deadlift():
    data = request.get_json()  # Retrieve JSON data from the request body
    frame_base64 = data.get('frame')  # Get the base64-encoded video frame from JSON

    # Convert base64 video data to numpy array
    frame_data = base64.b64decode(frame_base64)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

    # Save the received video as an MP4 file
    with open('received_video.mp4', 'wb') as f:
        f.write(frame_np)

    # Perform your pose correction or other processing on 'received_video.mp4'
    # Use the function from mediapipepose.py to process the video
    deadlift('received_video.mp4', 'processed_video.mp4')

    # Convert the processed video back to base64 string
    with open('processed_video.mp4', 'rb') as f:
        processed_video_data = f.read()

    processed_video_base64 = base64.b64encode(processed_video_data).decode('utf-8')

    # Return the processed video ('processed_video.mp4') as part of the JSON response
    return jsonify({'processed_video': processed_video_base64})

@app.route('/process_benchpress', methods=['POST'])
def process_benchpress():
    data = request.get_json()  # Retrieve JSON data from the request body
    frame_base64 = data.get('frame')  # Get the base64-encoded video frame from JSON

    # Convert base64 video data to numpy array
    frame_data = base64.b64decode(frame_base64)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

    # Save the received video as an MP4 file
    with open('received_video.mp4', 'wb') as f:
        f.write(frame_np)

    # Perform your pose correction or other processing on 'received_video.mp4'
    # Use the function from mediapipepose.py to process the video
    benchpress('received_video.mp4', 'processed_video.mp4')

    # Convert the processed video back to base64 string
    with open('processed_video.mp4', 'rb') as f:
        processed_video_data = f.read()

    processed_video_base64 = base64.b64encode(processed_video_data).decode('utf-8')

    # Return the processed video ('processed_video.mp4') as part of the JSON response
    return jsonify({'processed_video': processed_video_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
