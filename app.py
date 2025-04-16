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
import gc  # For garbage collection
import threading

# Lazy import YOLO to reduce startup memory usage
from functools import lru_cache

app = Flask(__name__)

# Configure CORS for production
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
model_lock = threading.Lock()  # Lock for thread-safe model access
MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', '1280'))  # Max image dimension
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', '85'))  # JPEG quality for processed images

# Class names for the model
CLASS_NAMES = ["Abiotic Disorder", "Cercospora", "Healthy", "Rust", "Sooty Mold"]
CLASS_COLORS = ["#f59e0b", "#b45309", "#dc2626", "#0891b2", "#4f46e5"]  # More colors

@lru_cache(maxsize=1)
def get_yolo():
    """Lazy import YOLO only when needed"""
    from ultralytics import YOLO
    return YOLO

def load_model(model_type="yolo8n"):
    """Load model with memory optimizations"""
    global model
    
    # For production, models should be in a consistent location
    base_path = os.environ.get('MODEL_PATH', 'weights')
    
    # Map frontend model selection to actual model path
    model_paths = {
        "yolo8n": f"{base_path}/yolov8n.pt",  # Nano model (smallest)
        "yolo8s": f"{base_path}/yolov8s.pt",  # Small model
        "yolo11m-full-leaf": f"{base_path}/yolo11m-full_leaf.pt",  # Original large model
    }
    
    model_path = model_paths.get(model_type, model_paths.get("yolo8n"))
    
    # Check if model file exists
    if not os.path.exists(model_path):
        app.logger.warning(f"Model file {model_path} not found")
        return False
    
    # Load the model with memory optimizations
    try:
        # Force garbage collection before loading model
        gc.collect()
        
        with model_lock:
            # Import YOLO only when needed
            YOLO = get_yolo()
            
            # Unload previous model if it exists
            if model is not None:
                del model
                gc.collect()
            
            # Load model with optimizations
            model = YOLO(model_path)
            
            # Disable model fusion to reduce memory usage
            if hasattr(model.model, 'fuse') and callable(model.model.fuse):
                model.model.fuse = lambda *args, **kwargs: model.model
            
            # Store the model type for reference
            setattr(model, 'model_type', model_type)
            
        app.logger.info(f"Successfully loaded model: {model_type}")
        return True
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess and resize image to reduce memory usage"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize large images to reduce memory usage
        width, height = image.size
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            app.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        return image_np, None
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {e}")
        return None, str(e)

# Initialize model on startup - use smaller model by default
load_model("yolo8n")  # Start with nano model to save memory

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    with model_lock:
        model_loaded = model is not None
        model_type = getattr(model, 'model_type', 'unknown') if model_loaded else 'none'
        
    mem_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = {
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent()
        }
    except ImportError:
        pass
        
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy", 
        "model_loaded": model_loaded,
        "model_type": model_type,
        "system_info": mem_info
    }), 200 if model_loaded else 503

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models"""
    base_path = os.environ.get('MODEL_PATH', 'weights')
    available_models = []
    
    model_info = {
        "yolo8n": {"name": "YOLOv8 Nano", "size": "Small (6MB)", "speed": "Fast"},
        "yolo8s": {"name": "YOLOv8 Small", "size": "Medium (20MB)", "speed": "Medium"},
        "yolo11m-full-leaf": {"name": "YOLO11m Full Leaf", "size": "Large (>100MB)", "speed": "Slow"}
    }
    
    for model_id, info in model_info.items():
        model_path = f"{base_path}/{model_id.replace('yolo8', 'yolov8')}.pt"
        if model_id == "yolo11m-full-leaf":
            model_path = f"{base_path}/yolo11m-full_leaf.pt"
            
        available = os.path.exists(model_path)
        available_models.append({
            "id": model_id,
            "name": info["name"],
            "size": info["size"],
            "speed": info["speed"],
            "available": available
        })
    
    return jsonify({"models": available_models})

@app.route('/api/detect', methods=['POST'])
def detect_disease():
    """Main detection endpoint with optimizations"""
    # Check if model is loaded
    with model_lock:
        if not model:
            app.logger.error("API called but model not loaded")
            return jsonify({"error": "Model not loaded"}), 500
    
    # Get parameters from request
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image data"}), 400
            
        image_data = data.get('image')
        model_type = data.get('modelType', 'yolo8n')  # Default to nano model
        detection_type = data.get('detectionType', 'disease')
        confidence_threshold = float(data.get('confidence', 50)) / 100
        overlap_threshold = float(data.get('overlap', 50)) / 100
    except Exception as e:
        app.logger.error(f"Error parsing request: {e}")
        return jsonify({"error": f"Invalid request format: {str(e)}"}), 400
    
    # Check if we need to load a different model
    with model_lock:
        current_model_type = getattr(model, 'model_type', None)
        
    if model_type != current_model_type:
        app.logger.info(f"Switching model to {model_type}")
        if not load_model(model_type):
            return jsonify({"error": f"Failed to load model: {model_type}"}), 500
    
    # Preprocess image
    image_np, error = preprocess_image(image_data)
    if error:
        return jsonify({"error": f"Error processing image: {error}"}), 400
    
    # Run detection with timeout handling
    try:
        start_time = time.time()
        
        with model_lock:
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
        
        # Create a copy of the image for drawing to avoid modifying the original
        image_with_boxes = image_np.copy()
        
        # Create processed image with bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = tuple(int(det["color"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to BGR
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Add text with better visibility
            text = f"{det['name']}: {det['confidence']}%"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_with_boxes, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image_with_boxes, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert back to RGB for frontend
        if len(image_with_boxes.shape) == 3 and image_with_boxes.shape[2] == 3:
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Encode processed image to base64 with quality control
        processed_image = Image.fromarray(image_with_boxes)
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=JPEG_QUALITY)
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        app.logger.info(f"Detection completed in {processing_time:.2f}s with {len(detections)} detections")
        
        # Force garbage collection after processing
        gc.collect()
        
        return jsonify({
            "detections": detections,
            "processedImage": f"data:image/jpeg;base64,{processed_image_base64}",
            "processingTime": round(processing_time, 2),
            "imageSize": {
                "width": image_np.shape[1],
                "height": image_np.shape[0]
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error during detection: {e}")
        # Force garbage collection after error
        gc.collect()
        return jsonify({"error": f"Error during detection: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)