from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure CORS
allowed_origins = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Set up logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Flask API startup')

# Global variables
model = None
model_loaded = False
model_loading = False

# Class names for the model
CLASS_NAMES = ["Abiotic Disorder", "Cercospora", "Healthy", "Rust", "Sooty Mold"]
CLASS_COLORS = ["#f59e0b", "#b45309", "#dc2626", "#0891b2", "#4f46e5"]

def load_model_in_background():
    """Load model in background to avoid blocking app startup"""
    global model, model_loaded, model_loading
    
    if model_loading:
        return False
    
    model_loading = True
    app.logger.info("Starting background model loading")
    
    try:
        # Import YOLO here to avoid loading at startup
        from ultralytics import YOLO
        
        # For production, models should be in a consistent location
        base_path = os.environ.get('MODEL_PATH', 'weights')
        
        # Use smaller model by default
        model_type = os.environ.get('DEFAULT_MODEL', 'yolov8n.pt')
        model_path = f"{base_path}/{model_type}"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            app.logger.warning(f"Model file {model_path} not found")
            model_loading = False
            return False
        
        # Load the model
        model = YOLO(model_path)
        model_loaded = True
        app.logger.info(f"Successfully loaded model: {model_type}")
        return True
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        return False
    finally:
        model_loading = False

# Start model loading in background
import threading
threading.Thread(target=load_model_in_background).start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model_loaded,
        "model_loading": model_loading
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect_disease():
    """Main detection endpoint"""
    global model, model_loaded
    
    # Check if model is loaded
    if not model_loaded:
        # Try loading model if not already loaded
        if not model_loading and not load_model_in_background():
            app.logger.error("API called but model not loaded")
            return jsonify({
                "error": "Model not loaded. The server is still initializing or encountered an error loading the model.",
                "status": "error",
                "modelLoaded": model_loaded,
                "modelLoading": model_loading
            }), 503  # Service Unavailable
    
    # Get parameters from request
    try:
        data = request.json
        image_data = data.get('image')
        confidence_threshold = float(data.get('confidence', 50)) / 100
        overlap_threshold = float(data.get('overlap', 50)) / 100
    except Exception as e:
        app.logger.error(f"Error parsing request: {e}")
        return jsonify({"error": f"Invalid request format: {str(e)}"}), 400
    
    # Decode base64 image
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Resize large images to reduce memory usage
        width, height = image.size
        max_size = 800
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            app.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    # Run detection
    try:
        start_time = time.time()
        
        results = model(image_np, conf=confidence_threshold, iou=overlap_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                
                # Calculate area percentage
                img_area = image_np.shape[0] * image_np.shape[1]
                box_area = (x2 - x1) * (y2 - y1)
                area_percentage = (box_area / img_area) * 100
                
                # Get color for this class
                color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
                
                # Add detection
                detections.append({
                    "name": class_name,
                    "confidence": round(confidence * 100, 2),
                    "area": round(area_percentage, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "color": color,
                    "description": f"Detected {class_name} with {round(confidence * 100, 2)}% confidence"
                })
        
        # Create processed image with bounding boxes
        image_with_boxes = image_np.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = tuple(int(det["color"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to BGR
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_with_boxes, f"{det['name']}: {det['confidence']}%", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB for frontend
        if len(image_with_boxes.shape) == 3 and image_with_boxes.shape[2] == 3:
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Encode processed image to base64
        processed_image = Image.fromarray(image_with_boxes)
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=85)
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        processing_time = time.time() - start_time
        app.logger.info(f"Detection completed in {processing_time:.2f}s with {len(detections)} detections")
        
        return jsonify({
            "detections": detections,
            "processedImage": f"data:image/jpeg;base64,{processed_image_base64}",
            "processingTime": round(processing_time, 2)
        })
        
    except Exception as e:
        app.logger.error(f"Error during detection: {e}")
        return jsonify({"error": f"Error during detection: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)