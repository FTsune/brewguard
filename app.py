from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model = None

def load_model(model_type="yolo11m-full-leaf"):
    global model
    
    # Map frontend model selection to actual model path
    model_paths = {
        "yolo11m-full-leaf": "weights/yolo11m-full_leaf.pt",  # Replace with your actual model path
        # "spots-full-leaf": "weights/yolov8s.pt"      # Replace with your actual model path
    }
    
    model_path = model_paths.get(model_type, model_paths["yolo11m-full-leaf"])
    
    # Load the model
    try:
        model = YOLO(model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize model on startup
load_model()

@app.route('/api/detect', methods=['POST'])
def detect_disease():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get parameters from request
    data = request.json
    image_data = data.get('image')
    model_type = data.get('modelType', 'yolo11m-full-leaf')
    detection_type = data.get('detectionType', 'disease')
    confidence_threshold = float(data.get('confidence', 50)) / 100
    overlap_threshold = float(data.get('overlap', 50)) / 100
    
    # Check if we need to load a different model
    if model_type != getattr(model, 'model_type', 'yolo11m-full-leaf'):
        load_model(model_type)
    
    # Decode base64 image
    try:
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    # Run detection
    try:
        # Simulate processing time for demo purposes
        time.sleep(1)
        
        results = model(image_np, conf=confidence_threshold, iou=overlap_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name (replace with your actual class names)
                class_names = ["Abiotic Disorder", "Cercospora", "Healthy", "Rust", "Sooty Mold"]
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                
                # Calculate area percentage (simplified)
                img_area = image_np.shape[0] * image_np.shape[1]
                box_area = (x2 - x1) * (y2 - y1)
                area_percentage = (box_area / img_area) * 100
                
                # Add detection
                detections.append({
                    "name": class_name,
                    "confidence": round(confidence * 100, 2),
                    "area": round(area_percentage, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "color": ["#f59e0b", "#b45309", "#dc2626"][class_id % 3],  # Cycle through colors
                    "description": f"Detected {class_name} with {round(confidence * 100, 2)}% confidence"
                })
        
        # Create processed image with bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = tuple(int(det["color"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to BGR
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, f"{det['name']}: {det['confidence']}%", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB for frontend
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Encode processed image to base64
        processed_image = Image.fromarray(image_np)
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "detections": detections,
            "processedImage": f"data:image/jpeg;base64,{processed_image_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error during detection: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)