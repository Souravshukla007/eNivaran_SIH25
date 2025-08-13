import os
import cv2
import numpy as np
import logging
import time
import json
import argparse
import torch
from collections import Counter, deque
from ultralytics import YOLO

# --- Configuration ---
# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pothole_detector.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Global model instance
_global_model = None

# --- Core Model and Logic Functions ---

def load_model(model_path):
    """
    Loads the YOLOv11 model using ultralytics YOLO.
    """
    global _global_model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading YOLOv11 model from {model_path}")
        
        # Load the model using ultralytics
        model = YOLO(model_path)
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Model will use device: {device}")
        
        # Move model to device
        model.to(device)
        
        _global_model = model
        logger.info("YOLOv11 model loaded successfully.")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 model: {e}", exc_info=True)
        raise

def get_model_device():
    """Get the current device of the global model."""
    global _global_model
    if _global_model is None:
        return 'cpu'
    return next(_global_model.parameters()).device

def move_model_to_device(device):
    """Move the global model to specified device."""
    global _global_model
    if _global_model is not None:
        _global_model.to(device)
        logger.info(f"Model moved to {device}")

def estimate_pothole_depth(image, contour):
    """
    Estimates pothole depth score (0-1) based on shadow analysis using contour.
    """
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        pixel_values = gray_image[mask == 255]
        if pixel_values.size == 0: 
            return 0.0
        
        darkness_score = 1 - (np.mean(pixel_values) / 255.0)
        contrast_score = min(np.std(pixel_values) / 50.0, 1.0) if pixel_values.size > 1 else 0.0
        return max(0.0, min(1.0, (0.7 * darkness_score) + (0.3 * contrast_score)))
        
    except Exception as e:
        logger.warning(f"Could not estimate depth: {e}")
        return 0.0

def get_individual_pothole_priority(area_ratio, depth_score):
    """
    Determines an individual pothole's priority ('High', 'Medium', 'Low').
    """
    combined_score = (0.6 * area_ratio * 100) + (0.4 * depth_score)
    if combined_score > 0.6 or (area_ratio > 0.01 and depth_score > 0.6):
        return 'High', (0, 0, 255)  # Red
    elif combined_score > 0.3 or (area_ratio > 0.005 and depth_score > 0.4):
        return 'Medium', (0, 165, 255)  # Orange
    else:
        return 'Low', (0, 255, 0)  # Green

def determine_road_priority(potholes_list, proximity_threshold, image_shape):
    """
    Determines the overall road priority based on all detected potholes.
    """
    if not potholes_list:
        return 'Low', (0, 255, 0), []
    
    high_count = sum(1 for p in potholes_list if p['priority'] == 'High')
    medium_count = sum(1 for p in potholes_list if p['priority'] == 'Medium')
    
    clusters, processed = [], set()
    for i, p1 in enumerate(potholes_list):
        if i in processed: 
            continue
        cluster, q = [i], [i]
        processed.add(i)
        while q:
            curr = q.pop(0)
            for j, p2 in enumerate(potholes_list):
                if j not in processed and np.linalg.norm(np.array(p1['position']) - np.array(p2['position'])) < proximity_threshold:
                    processed.add(j)
                    cluster.append(j)
                    q.append(j)
        clusters.append(cluster)
    
    total_area_ratio = sum(p['area_ratio'] for p in potholes_list)
    
    if (high_count >= 2 or (high_count >= 1 and medium_count >= 2) or
        total_area_ratio > 0.05 or len([c for c in clusters if len(c) >= 3]) > 0):
        return 'High', (0, 0, 255), clusters
    elif (high_count >= 1 or medium_count >= 2 or total_area_ratio > 0.02 or 
          len([c for c in clusters if len(c) >= 2]) > 0):
        return 'Medium', (0, 165, 255), clusters
    else:
        return 'Low', (0, 255, 0), clusters

# --- Main Assessment Function for Images ---

def assess_road_image(image_path, model, conf_threshold=0.25, proximity_threshold=150, device='cpu'):
    """
    Assesses a single image, returning a JSON report and an annotated image.
    """
    try:
        # Ensure model is on the correct device
        current_device = str(get_model_device())
        if device != current_device:
            move_model_to_device(device)
        
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:  # Assumes image_path is a numpy array
            image = image_path

        annotated_image = image.copy()
        h, w = image.shape[:2]
        image_area = h * w
        
        # Run inference with error handling
        try:
            results = model(image, conf=conf_threshold, device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'cuda' in str(e).lower() and device == 'cuda':
                logger.warning(f"CUDA error during inference, falling back to CPU: {e}")
                move_model_to_device('cpu')
                results = model(image, conf=conf_threshold, device='cpu')
            else:
                raise
        
        result = results[0]
        potholes_list = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            # Get contours from masks if available, otherwise create from bounding boxes
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks.xy) > 0:
                contours = [c.astype(int) for c in result.masks.xy]
            else:
                # Fallback to using bounding boxes as contours if no masks
                contours = [
                    np.array([
                        [int(box[0]), int(box[1])], [int(box[2]), int(box[1])],
                        [int(box[2]), int(box[3])], [int(box[0]), int(box[3])]
                    ], dtype=np.int32) for box in boxes
                ]

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                confidence = float(confidences[i])
                
                # Use the corresponding contour
                contour = contours[i]
                
                # Calculate actual area from contour
                contour_area = cv2.contourArea(contour)
                area_ratio = contour_area / image_area
                
                depth_score = estimate_pothole_depth(image, contour)
                priority, color = get_individual_pothole_priority(area_ratio, depth_score)
                
                potholes_list.append({
                    'id': i, 
                    'position': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'bbox': [x1, y1, x2, y2], 
                    'area_ratio': area_ratio,
                    'area_pixels': int(contour_area),
                    'depth_score': depth_score, 
                    'priority': priority, 
                    'confidence': confidence,
                    'contour': contour.tolist()  # Store contour points
                })
                
                # Draw contour instead of rectangle for cleaner look
                cv2.drawContours(annotated_image, [contour], -1, color, 2)
                
                # Also draw bounding box with dashed line style
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                
                # Add label
                cv2.putText(annotated_image, f"{priority} ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        road_priority, road_color, clusters = determine_road_priority(potholes_list, proximity_threshold, (h, w))
        
        # Draw cluster connections
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                points = np.array([potholes_list[idx]['position'] for idx in cluster]).astype(np.int32)
                hull = cv2.convexHull(points.reshape(-1, 1, 2))
                cv2.polylines(annotated_image, [hull], True, (255, 0, 255), 2)
        
        # Add road priority text
        cv2.putText(annotated_image, f"Road Priority: {road_priority}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, road_color, 3)
        
        priority_counts = Counter(p['priority'] for p in potholes_list)
        source_name = os.path.basename(image_path) if isinstance(image_path, str) else "image_from_memory"

        # Remove contour data from JSON output (too large)
        json_potholes = [{k: p[k] for k in ('id', 'bbox', 'priority', 'depth_score', 'confidence', 'area_ratio', 'area_pixels')} 
                        for p in potholes_list]

        assessment_data = {
            "source": source_name, 
            "road_priority": road_priority,
            "total_potholes": len(potholes_list),
            "priority_distribution": dict(priority_counts),
            "cluster_count": len([c for c in clusters if len(c) > 1]),
            "total_pothole_area_pixels": sum(p['area_pixels'] for p in potholes_list),
            "total_area_ratio": sum(p['area_ratio'] for p in potholes_list),
            "potholes": json_potholes
        }
        
        return json.dumps(assessment_data, indent=2), annotated_image
        
    except Exception as e:
        logger.error(f"Error in assess_road_image: {e}", exc_info=True)
        raise

# --- Video Assessment Function with Batching ---

def assess_road_video(video_path, output_path, model_path='Pothole-Detector.pt', batch_size=16):
    """
    Assesses a video with GPU acceleration and frame batching.
    """
    try:
        # Load the model
        model = load_model(model_path)
        
        # Determine device for video processing (prefer GPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Video processing will use device: {device}")
        print(f"--- Pothole Detection: Video processing is using {device.upper()} ---")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
        
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_count = 0
        frame_batch = []
        processed_frames = []
        
        logger.info(f"Processing {total_frames} frames with batch size {batch_size}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in batch
                if frame_batch:
                    processed_batch = process_frame_batch(frame_batch, model, device)
                    processed_frames.extend(processed_batch)
                break
            
            frame_batch.append(frame.copy())
            frame_count += 1
            
            # Process batch when it's full
            if len(frame_batch) >= batch_size:
                try:
                    processed_batch = process_frame_batch(frame_batch, model, device)
                    processed_frames.extend(processed_batch)
                    
                    # Write processed frames
                    for proc_frame in processed_frames:
                        out.write(proc_frame)
                    
                    # Clear batches
                    frame_batch = []
                    processed_frames = []
                    
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
                    
                except Exception as e:
                    logger.error(f"Error processing batch at frame {frame_count}: {e}")
                    # Fallback: process frames individually on CPU
                    for frame in frame_batch:
                        _, annotated_frame = assess_road_image(frame, model, device='cpu')
                        out.write(annotated_frame)
                    frame_batch = []
        
        # Write any remaining processed frames
        for proc_frame in processed_frames:
            out.write(proc_frame)

        cap.release()
        out.release()
        
        logger.info(f"Video processing complete. Output saved to {output_path}")
        return output_path, 0  # Return 0 for avg_damage as it's no longer calculated

    except Exception as e:
        logger.error(f"Error in assess_road_video: {e}", exc_info=True)
        if 'cap' in locals() and cap.isOpened(): 
            cap.release()
        if 'out' in locals(): 
            out.release()
        return None, 0

def process_frame_batch(frame_batch, model, device):
    """
    Process a batch of frames efficiently on GPU.
    """
    try:
        processed_frames = []
        
        # For now, process frames individually but on the specified device
        # Batched inference can be complex with bounding box outputs
        for frame in frame_batch:
            _, annotated_frame = assess_road_image(frame, model, device=device)
            processed_frames.append(annotated_frame)
        
        return processed_frames
        
    except Exception as e:
        logger.error(f"Error in process_frame_batch: {e}")
        # Fallback to CPU processing
        processed_frames = []
        for frame in frame_batch:
            _, annotated_frame = assess_road_image(frame, model, device='cpu')
            processed_frames.append(annotated_frame)
        return processed_frames

# --- Flask Integration Functions ---

def run_pothole_detection(image_path):
    """
    Flask-ready entry point for pothole detection from file path.
    Uses CPU for web requests.
    """
    try:
        model = load_model("Pothole-Detector.pt")
        json_output, annotated_image = assess_road_image(image_path, model, device='cpu')
        
        result_dict = json.loads(json_output)
        result_dict['image_path'] = image_path
        
        success, img_encoded = cv2.imencode('.jpg', annotated_image)
        if not success:
            return None, None
            
        return result_dict, img_encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Error in run_pothole_detection: {e}", exc_info=True)
        return None, None

def run_pothole_detection_from_bytes(image_bytes):
    """
    Flask-ready entry point for pothole detection from image bytes.
    Uses CPU for web requests.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
            
        model = load_model("Pothole-Detector.pt")
        json_output, annotated_image = assess_road_image(img, model, device='cpu')
        
        result_dict = json.loads(json_output)
        
        success, img_encoded = cv2.imencode('.jpg', annotated_image)
        if not success:
            return None, None
            
        return result_dict, img_encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Error in run_pothole_detection_from_bytes: {e}", exc_info=True)
        return None, None

if __name__ == "_main_":
    parser = argparse.ArgumentParser(description="Pothole Detection and Road Priority Assessment.")
    parser.add_argument("--image", type=str, help="Path to a single image for analysis.")
    parser.add_argument("--video", type=str, help="Path to a video file for analysis.")
    parser.add_argument("--output", type=str, help="Output path for processed video.")
    parser.add_argument("--model", type=str, default="Pothole-Detector.pt", help="Path to the YOLOv11 model file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                        help="Device to use for inference.")

    args = parser.parse_args()

    try:
        # Determine device
        if args.device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        model = load_model(args.model)
        
        if args.image:
            if not os.path.exists(args.image):
                logger.error(f"Image file not found: {args.image}")
            else:
                logger.info(f"--- Processing Image: {args.image} ---")
                json_output, annotated_image = assess_road_image(args.image, model, 
                                                               conf_threshold=args.conf, device=device)
                
                output_path = f"{os.path.splitext(args.image)[0]}_assessed.jpg"
                cv2.imwrite(output_path, annotated_image)
                
                print("\n--- Assessment Report ---")
                print(json_output)
                logger.info(f"Annotated image saved to: {output_path}")
        
        elif args.video:
            if not os.path.exists(args.video):
                logger.error(f"Video file not found: {args.video}")
            else:
                if not args.output:
                    args.output = f"{os.path.splitext(args.video)[0]}_assessed.mp4"
                
                logger.info(f"--- Processing Video: {args.video} ---")
                output_path, _ = assess_road_video(args.video, args.output, args.model)
                if output_path:
                    logger.info(f"Processed video saved to: {output_path}")
                else:
                    logger.error("Video processing failed")
        
        else:
            parser.print_help()
            logger.warning("No input file specified. Use --image or --video arguments.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
