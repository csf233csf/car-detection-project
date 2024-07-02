from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
    
# ort_session = ort.InferenceSession('checkpoint/train3/weights/best.onnx')

# # Print input names
# input_name = ort_session.get_inputs()[0].name
# print(f"Model input name: {input_name}")

# exit()

app = Flask(__name__)

ort_session = ort.InferenceSession('checkpoint/train3/weights/best.onnx')

def preprocess_image(image):
    image = image.resize((640, 640))
    image = np.array(image).astype('float32')
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file)
    input_data = preprocess_image(image)
    
    outputs = ort_session.run(None, {'images': input_data})
    print(outputs[0])
    result = []
    for output in outputs[0]:
        x1, y1, x2, y2, score = output
        result.append({
            'bbox': [x1, y1, x2, y2],
            'score': float(score),
            'label': 'car'
        })
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
