from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from imread_from_url import imread_from_url
from yolov8 import YOLOv8
from PIL import Image
import io

app = Flask(__name__)

# Initialize yolov8 object detector
model_path = "checkpoint/train3/weights/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the image file
    image_bytes = file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Detect Objects
    boxes, scores, class_ids = yolov8_detector(img)
    
    # Draw detections
    combined_img = yolov8_detector.draw_detections(img)
    
    # Convert image to byte array for response
    is_success, buffer = cv2.imencode(".jpg", combined_img)
    io_buf = io.BytesIO(buffer)
    
    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
