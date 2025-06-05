from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from werkzeug.utils import secure_filename

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MediaPipe Pose
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    """Calculate angle (in degrees) at point b given points a, b, and c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect pose
        detection_result = detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return jsonify({'error': 'No pose detected'}), 400

        # Get landmarks and calculate angle
        landmarks = detection_result.pose_landmarks[0]
        elbow = landmarks[14]
        shoulder = landmarks[12]
        hip = landmarks[24]

        # Calculate coordinates
        image_height, image_width, _ = image.shape
        elbow_coord = (elbow.x * image_width, elbow.y * image_height)
        shoulder_coord = (shoulder.x * image_width, shoulder.y * image_height)
        hip_coord = (hip.x * image_width, hip.y * image_height)

        # Calculate angle
        shoulder_angle = calculate_angle(elbow_coord, shoulder_coord, hip_coord)
        
        # Draw landmarks
        annotated_image = draw_landmarks_on_image(image, detection_result)
        
        # Save annotated image
        output_filename = f'annotated_{filename}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_image)

        # Determine state
        state = "Safe" if shoulder_angle <= 100 else "Unsafe"

        return jsonify({
            'angle': round(shoulder_angle, 2),
            'state': state,
            'image_url': f'/static/uploads/{output_filename}'
        })

if __name__ == '__main__':
    app.run(debug=True) 