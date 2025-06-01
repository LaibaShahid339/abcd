from flask import Flask, request, send_file
import cv2
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
frozen_model = "frozen_inference_graph.pb"
file_names = "labels.txt"
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

with open(file_names, "rt") as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

@app.route('/object-detection', methods=['POST'])
def object_detection():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    if ClassIndex is not None and len(ClassIndex) > 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, boxes, color=(255, 0, 0), thickness=2)
            cv2.putText(img, class_labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        text = 'No Object Detected'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 3
        box_color = (0, 0, 0)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        position = (img.shape[1] - text_width - 10, text_height + 10)
        cv2.rectangle(img, (position[0] - 10, position[1] - text_height - 10),
                      (position[0] + text_width + 10, position[1] + 10), box_color, cv2.FILLED)
        cv2.putText(img, text, position, font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(buffer)
    return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
