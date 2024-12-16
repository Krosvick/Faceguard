import threading
import os
import json
from pathlib import Path
from datetime import datetime
import queue
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from typing import Dict, Optional, List
import yaml
import time
from torchvision import transforms
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
import requests
from urllib.error import URLError
import backoff
from queue import Queue
from threading import Thread
import signal

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (choose one)
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device)

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Mapping of face IDs to names
id_face_mapping = {}

# Data mapping for tracking information
data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}

# Initialize a lock
data_lock = threading.Lock()

# Initialize queues
tracking_queue = queue.Queue(maxsize=30)  # Increased from 10

ROOM_ID = "LC6"
DISPLAY_CONFIDENCE_THRESHOLD = 0.25  # Threshold for displaying names
RECORD_CONFIDENCE_THRESHOLD = 0.6    # Higher threshold for recording attendance
RECOGNITION_COOLDOWN = 300  # 5 minutes between recognitions of the same person
last_recognition_time: Dict[str, datetime] = {}
best_recognition: Dict[str, Optional[dict]] = {}
processed_names = set()  # Track which names have been processed
recognition_results = {}  # Store recognition results

RECORDS_DIR = Path("./attendance_records")
PENDING_RECORDS_FILE = RECORDS_DIR / "pending_records.json"
BATCH_SIZE = 30  # Increased from 10

# Initialize records directory
RECORDS_DIR.mkdir(exist_ok=True)

API_BASE_URL = "http://localhost:3000/api"  # Change this to your Next.js API URL
SEND_INTERVAL = 5  # Seconds between sending batches to API
MAX_RETRIES = 3

# Add new variable to track class end times
class_end_times = {}

api_queue = Queue()
SHUTDOWN_FLAG = False

# Constants for recognition
UNKNOWN_TAG = "Desconocido"
HTTP_TRIGGER_THRESHOLD = 0.55  # Threshold for triggering HTTP requests
UNKNOWN_THRESHOLD = 0.20  # Threshold below which faces are labeled as unknown
UNKNOWN_FEATURE_SIMILARITY_THRESHOLD = 0.25  # Lower threshold for matching unknown faces
UNKNOWN_FEATURES_EXPIRY = 3600  # 1 hour expiry for unknown features
MIN_FEATURES_FOR_MATCH = 3  # Minimum number of similar features to consider a match

# Track best quality frames and last recognition times
best_quality_frames = {}  # {name: {"quality": float, "image": str, "timestamp": datetime}}
last_recognition_times = {}  # {name: datetime}

# Track unknown individuals
unknown_features = {}  # {temp_id: {"features": np.array, "last_seen": datetime, "count": int}}
next_unknown_id = 1  # Counter for generating temporary IDs

def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def signal_handler(sig, frame):
    global SHUTDOWN_FLAG
    print("Shutdown signal received. Exiting gracefully...")
    SHUTDOWN_FLAG = True

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def process_tracking(frame, detector, tracker, args, frame_id, fps):
    try:
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
        tracking_tlwhs = []
        tracking_ids = []
        tracking_scores = []
        tracking_bboxes = []

        # Only reset tracker if no detections for multiple consecutive frames
        if outputs is None:
            if not hasattr(tracker, 'empty_frames'):
                tracker.empty_frames = 0
            tracker.empty_frames += 1
            if tracker.empty_frames > 10:  # Reset after 10 empty frames
                tracker.empty_frames = 0
                tracker.reset_track()
            return img_info["raw_img"]

        tracker.empty_frames = 0  # Reset counter when detections found

        # Update tracker with frame timestamp
        online_targets = tracker.update(
            outputs, 
            [img_info["height"], img_info["width"]], 
            (128, 128)
        )

        if online_targets is not None:
            for i in range(len(online_targets)):
                t = online_targets[i]
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
                if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                    x1, y1, w, h = tlwh
                    tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
                    tracking_scores.append(t.score)

            tracking_image = plot_tracking(
                img_info["raw_img"],
                tracking_tlwhs,
                tracking_ids,
                names=id_face_mapping,
                frame_id=frame_id + 1,
                fps=fps,
            )
        else:
            tracking_image = img_info["raw_img"]

        # Prepare tracking data
        tracking_data = {
            "raw_image": img_info["raw_img"],
            "detection_bboxes": bboxes,
            "detection_landmarks": landmarks,
            "tracking_ids": tracking_ids,
            "tracking_bboxes": tracking_bboxes
        }

        try:
            tracking_queue.put_nowait(tracking_data)
        except queue.Full:
            print("Tracking queue is full. Dropping frame.")

        return tracking_image
        
    except Exception as e:
        print(f"Error in process_tracking: {e}")
        return frame  # Return original frame if processing fails

@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Inference to get feature
    emb_img_face = recognizer(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb

def recognition(face_image):
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]
    return score, name

def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou

def tracking(detector, args):
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    last_frame_time = time.time()
    frame_timeout = 1.0  # Maximum time to wait for a frame

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(1)
    
    # Apply camera optimizations
    optimize_camera_settings(cap)
    
    # Create named window with proper flags
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while not SHUTDOWN_FLAG:
        current_time = time.time()
        
        # Read frame with timeout
        ret = False
        frame_attempts = 0
        max_attempts = 3
        
        while not ret and frame_attempts < max_attempts:
            ret, img = cap.read()
            if not ret:
                frame_attempts += 1
                print(f"Failed to grab frame, attempt {frame_attempts}/{max_attempts}")
                # Flush buffer
                cap.grab()
                time.sleep(0.1)
        
        if not ret:
            print("Camera connection may be lost, attempting to reconnect...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(1)
            optimize_camera_settings(cap)
            continue

        # Update last frame time
        last_frame_time = current_time

        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time_ns()
            fps = 1e9 * frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time_ns()

        # Only update display if window is visible
        try:
            cv2.imshow("Face Recognition", tracking_image)
        except cv2.error:
            print("Display error, continuing processing...")

        # Check for user exit input with a short timeout
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        # Small sleep to prevent CPU overload when in background
        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()

def save_face_image(face_image, name: str, quality: float):
    """Save the face image with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recognized_faces/{name}_{timestamp}_{quality:.2f}.jpg"
    cv2.imwrite(filename, face_image)
    return filename

def encode_image_base64(face_image):
    """Convert CV2 image to base64 string."""
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Create a BytesIO object
    buffered = BytesIO()

    # Save image to BytesIO object as JPEG
    pil_image.save(buffered, format="JPEG", quality=95)

    # Encode as base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

def load_pending_records():
    """Load pending records from JSON file."""
    if PENDING_RECORDS_FILE.exists():
        try:
            with open(PENDING_RECORDS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_pending_records(records):
    """Save pending records to JSON file with error handling."""
    try:
        existing_records = load_pending_records()
        # Add new records, avoiding duplicates based on timestamp
        existing_timestamps = {r["timestamp"] for r in existing_records}
        new_records = [r for r in records if r["timestamp"] not in existing_timestamps]
        all_records = existing_records + new_records
        
        # Ensure directory exists
        RECORDS_DIR.mkdir(exist_ok=True)
        
        # Write atomically using temporary file
        temp_file = PENDING_RECORDS_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(all_records, f, indent=2)
        temp_file.replace(PENDING_RECORDS_FILE)
        
        if new_records:
            print(f"Saved {len(new_records)} new records to pending file")
    except Exception as e:
        print(f"Error saving pending records: {e}")

def clear_sent_records(successful_records):
    """Remove successfully sent records from pending records."""
    pending_records = load_pending_records()
    remaining_records = [r for r in pending_records 
                        if r["timestamp"] not in [sr["timestamp"] for sr in successful_records]]
    save_pending_records(remaining_records)

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, URLError),
    max_tries=5,  # Maximum number of retries
    max_time=300,  # Maximum total time to try (5 minutes)
    on_backoff=lambda details: print(f"Retry attempt {details['tries']} after {details['wait']:.1f} seconds")
)
def send_records_to_api(records):
    """
    Send records to Next.js API with retry logic.
    Returns successfully sent records or empty list if all failed.
    """
    if not records:
        return []
        
    try:
        response = requests.post(
            f"{API_BASE_URL}/attendance/record",
            json={"records": records},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('classEndTime'):
                end_time = datetime.fromisoformat(data['classEndTime'].replace('Z', '+00:00'))
                room = records[0]['room']
                class_end_times[room] = end_time.replace(tzinfo=None)
            print(f"Successfully sent {len(records)} records to API")
            return records
        else:
            print(f"Failed to send records. Status: {response.status_code}")
            raise requests.exceptions.RequestException(f"API returned status code {response.status_code}")
            
    except Exception as e:
        print(f"Error sending records to API: {e}")
        raise  # Re-raise the exception for backoff to handle

def should_record_attendance(name: str, room: str, current_time: datetime) -> bool:
    """
    Determine if attendance should be recorded for a person.
    """
    # If person hasn't been processed yet, allow recording
    if name not in processed_names:
        return True
        
    # If we have a class end time for this room
    if room in class_end_times:
        # Ensure current_time is timezone-naive for comparison
        current_time_naive = current_time.replace(tzinfo=None)
        # Allow new recording if current time is past the class end time
        if current_time_naive > class_end_times[room]:
            # Remove from processed names to allow new recording
            processed_names.remove(name)
            return True
            
    return False

def api_worker():
    """Worker thread to handle API requests with improved retry logic."""
    retry_queue = Queue()  # Queue for failed requests that need retry
    
    while not SHUTDOWN_FLAG:
        try:
            # First try to process any retried records
            while not retry_queue.empty() and not SHUTDOWN_FLAG:
                retry_records = retry_queue.get()
                try:
                    successful_records = send_records_to_api(retry_records)
                    if successful_records:
                        clear_sent_records(successful_records)
                        print(f"Successfully retried sending {len(successful_records)} records")
                except Exception as e:
                    print(f"Retry failed, will try again later: {e}")
                    save_pending_records(retry_records)  # Save to disk for persistence
                    time.sleep(5)  # Wait before next retry
            
            # Process new records
            try:
                records = api_queue.get_nowait()
                
                if records is None:  # Shutdown signal
                    break
                    
                try:
                    successful_records = send_records_to_api(records)
                    if successful_records:
                        clear_sent_records(successful_records)
                except Exception as e:
                    print(f"Error sending records to API: {e}")
                    retry_queue.put(records)  # Add to retry queue
                    save_pending_records(records)  # Save to disk for persistence
                    
            except queue.Empty:
                # Load any pending records from disk and try to send them
                pending_records = load_pending_records()
                if pending_records:
                    try:
                        successful_records = send_records_to_api(pending_records)
                        if successful_records:
                            clear_sent_records(successful_records)
                            print(f"Successfully sent pending records: {len(successful_records)}")
                    except Exception as e:
                        print(f"Failed to send pending records: {e}")
                        time.sleep(5)  # Wait before next attempt
                else:
                    time.sleep(0.1)  # Small sleep to prevent CPU spinning
                    
        except Exception as e:
            print(f"Error in API worker: {e}")
            time.sleep(1)  # Brief pause before continuing

    # On shutdown, save any remaining records
    try:
        while not api_queue.empty():
            records = api_queue.get_nowait()
            if records is not None:  # Skip shutdown signal
                save_pending_records(records)
        while not retry_queue.empty():
            records = retry_queue.get_nowait()
            save_pending_records(records)
    except Exception as e:
        print(f"Error saving remaining records during shutdown: {e}")

def should_trigger_recognition(name: str, current_time: datetime) -> bool:
    """Determine if we should trigger a new recognition for this person."""
    if name not in last_recognition_times:
        return True
        
    time_diff = (current_time - last_recognition_times[name]).total_seconds()
    return time_diff >= RECOGNITION_COOLDOWN

def update_best_quality_frame(name: str, face_img, quality: float, current_time: datetime):
    """Update the best quality frame for a person if current quality is better."""
    if name not in best_quality_frames or quality > best_quality_frames[name]["quality"]:
        face_image_base64 = encode_image_base64(face_img)
        best_quality_frames[name] = {
            "quality": quality,
            "image": face_image_base64,
            "timestamp": current_time
        }

def normalize_features(features):
    """Normalize feature vector for more robust comparison."""
    return features / (np.linalg.norm(features) + 1e-6)

def compare_face_features(features1, features2):
    """
    Compare face features using multiple similarity metrics.
    Returns a similarity score between 0 and 1.
    """
    # Normalize features
    features1_norm = normalize_features(features1.flatten())
    features2_norm = normalize_features(features2.flatten())
    
    # Cosine similarity
    cosine_sim = np.dot(features1_norm, features2_norm)
    
    # Euclidean distance (converted to similarity)
    euclidean_dist = np.linalg.norm(features1_norm - features2_norm)
    euclidean_sim = 1 / (1 + euclidean_dist)
    
    # Combine similarities (you can adjust weights)
    combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
    
    return combined_sim

def get_temp_unknown_id(face_features: np.ndarray, current_time: datetime) -> str:
    """
    Get a temporary ID for an unknown face by comparing with stored unknown features.
    Uses multiple similarity metrics and temporal consistency.
    """
    global next_unknown_id
    
    # Clean up expired unknown features
    expired = []
    for temp_id, data in unknown_features.items():
        if (current_time - data["last_seen"]).total_seconds() > UNKNOWN_FEATURES_EXPIRY:
            expired.append(temp_id)
    for temp_id in expired:
        del unknown_features[temp_id]
    
    # Compare with existing unknown features
    best_match_id = None
    best_match_score = 0
    recent_seen_bonus = 0.1  # Bonus for recently seen faces
    
    for temp_id, data in unknown_features.items():
        # Calculate base similarity
        similarity = compare_face_features(face_features, data["features"])
        
        # Add bonus for recently seen faces (within last 30 seconds)
        time_diff = (current_time - data["last_seen"]).total_seconds()
        if time_diff < 30:
            similarity += recent_seen_bonus * (1 - time_diff/30)
        
        # Update best match if this is better
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_id = temp_id
    
    # If we found a good match, update it
    if best_match_score > UNKNOWN_FEATURE_SIMILARITY_THRESHOLD:
        data = unknown_features[best_match_id]
        # Update with moving average of features
        alpha = 0.3  # Weight for new features
        data["features"] = (1 - alpha) * data["features"] + alpha * face_features
        data["features"] = normalize_features(data["features"])
        data["last_seen"] = current_time
        data["count"] += 1
        return f"{UNKNOWN_TAG}_{best_match_id}"
    
    # If no match found, create new temporary ID
    temp_id = next_unknown_id
    next_unknown_id += 1
    unknown_features[temp_id] = {
        "features": normalize_features(face_features),
        "last_seen": current_time,
        "count": 1
    }
    return f"{UNKNOWN_TAG}_{temp_id}"

def recognize():
    """Face recognition in a separate thread."""
    global SHUTDOWN_FLAG
    last_send_time = time.time()
    global processed_names, recognition_results
    pending_records = load_pending_records()
    batch_count = 0

    # Start API worker thread
    api_thread = Thread(target=api_worker, daemon=True)
    api_thread.start()

    try:
        while not SHUTDOWN_FLAG:
            try:
                tracking_data = tracking_queue.get(timeout=1)
            except queue.Empty:
                if pending_records:
                    api_queue.put(pending_records)
                    pending_records = []
                if SHUTDOWN_FLAG:
                    break
                continue

            try:
                raw_image = tracking_data["raw_image"]
                detection_landmarks = tracking_data["detection_landmarks"]
                detection_bboxes = tracking_data["detection_bboxes"]
                tracking_ids = tracking_data["tracking_ids"]
                tracking_bboxes = tracking_data["tracking_bboxes"]

                faces_to_recognize = []
                ids_to_update = []
                quality_scores = []
                face_images = []
                current_time = datetime.now()

                for i in range(len(tracking_bboxes)):
                    for j in range(len(detection_bboxes)):
                        mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                        if mapping_score > 0.9:
                            face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])
                            is_quality_ok, quality_score = check_face_quality(face_alignment)
                            if is_quality_ok:
                                faces_to_recognize.append(face_alignment)
                                ids_to_update.append(tracking_ids[i])
                                quality_scores.append(quality_score)
                                face_images.append(face_alignment)
                            break

                if faces_to_recognize:
                    features = get_feature_batch(faces_to_recognize)
                    scores, names = compare_encodings_batch(features, images_embs)
                    for idx, (score, name, tid, quality, face_img) in enumerate(zip(scores, names, ids_to_update, quality_scores, face_images)):
                        # Determine display name based on confidence
                        if score <= UNKNOWN_THRESHOLD:
                            # Get or assign temporary ID for unknown face
                            display_name = get_temp_unknown_id(features[idx], current_time)
                        elif score >= DISPLAY_CONFIDENCE_THRESHOLD:
                            display_name = name
                        else:
                            # For faces between UNKNOWN_THRESHOLD and DISPLAY_CONFIDENCE_THRESHOLD
                            display_name = get_temp_unknown_id(features[idx], current_time)
                            
                        caption = f"{display_name}:{score:.2f}"
                        
                        # Update best quality frame regardless of recognition trigger
                        update_best_quality_frame(display_name, face_img, quality, current_time)
                        
                        # Trigger HTTP request if confidence meets threshold or it's an unknown face
                        if (score >= HTTP_TRIGGER_THRESHOLD or display_name.startswith(UNKNOWN_TAG)) and should_trigger_recognition(display_name, current_time):
                            # Use the best quality frame for this person
                            best_frame = best_quality_frames[display_name]
                            
                            record = {
                                "name": display_name,
                                "confidence": float(score),
                                "quality": float(best_frame["quality"]),
                                "timestamp": current_time.isoformat(),
                                "room": ROOM_ID,
                                "image": best_frame["image"],
                                "status": "pending"
                            }
                            
                            recognition_results[display_name] = record
                            pending_records.append(record)
                            last_recognition_times[display_name] = current_time
                            
                            batch_count += 1
                            if batch_count >= BATCH_SIZE:
                                save_pending_records(pending_records)
                                try:
                                    api_queue.put_nowait(pending_records)
                                except queue.Full:
                                    print("API queue full, saving to pending records")
                                    save_pending_records(pending_records)
                                pending_records = []
                                batch_count = 0
                                
                            # Print the recognition record (excluding the base64 image)
                            print_record = {**record}
                            print_record["image"] = "<<base64_image_data>>"
                            print("New Recognition:", json.dumps(print_record, indent=2))
                        
                        id_face_mapping[tid] = caption

                if not tracking_bboxes:
                    print("Waiting for a person...")

                # After processing new recognitions, check if it's time to send
                current_time = time.time()
                if pending_records and (current_time - last_send_time) >= SEND_INTERVAL:
                    try:
                        api_queue.put_nowait(pending_records)
                        pending_records = []
                        last_send_time = current_time
                    except queue.Full:
                        print("API queue full, will retry later")

            except Exception as e:
                print(f"Error in recognition thread: {e}")
                if pending_records:
                    save_pending_records(pending_records)

    except Exception as e:
        print(f"Error in recognition thread: {e}")
        if pending_records:
            save_pending_records(pending_records)
    finally:
        SHUTDOWN_FLAG = True
        api_queue.put(None)  # Signal the API worker to shut down
        api_thread.join(timeout=5)  # Wait for API worker to finish

@torch.no_grad()
def get_feature_batch(face_images):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Process faces in smaller sub-batches if needed
    batch_size = 32  # Adjust based on your GPU memory
    all_embeddings = []
    
    for i in range(0, len(face_images), batch_size):
        batch = face_images[i:i + batch_size]
        processed_faces = [face_preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]
        processed_faces = torch.stack(processed_faces).to(device)
        
        # Get embeddings for this sub-batch
        emb_imgs = recognizer(processed_faces).cpu().numpy()
        all_embeddings.append(emb_imgs)
    
    # Combine all embeddings
    emb_imgs = np.concatenate(all_embeddings, axis=0)
    
    # Normalize
    images_embs = emb_imgs / np.linalg.norm(emb_imgs, axis=1, keepdims=True)
    
    return images_embs

def compare_encodings_batch(features, reference_features):
    similarities = np.dot(features, reference_features.T)
    scores = similarities.max(axis=1)
    ids_min = similarities.argmax(axis=1)
    return scores, [images_names[idx] for idx in ids_min]

def check_face_quality(face_image):
    """Enhanced quality check with efficient blur detection"""
    # Calculate basic quality metrics
    brightness = np.mean(face_image)
    contrast = np.std(face_image)
    height, width = face_image.shape[:2]
    min_size = min(height, width)
    
    # Efficient blur detection using Laplacian
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Weighted quality score
    quality_score = 0.0
    quality_score += 0.2 * (1 if brightness > 40 and brightness < 250 else 0)
    quality_score += 0.2 * (1 if contrast > 20 else 0)
    quality_score += 0.2 * (1 if min_size >= 112 else 0)
    quality_score += 0.4 * (1 if blur_score > 100 else 0)  # Increased weight for blur
    
    return quality_score > 0.6, quality_score

def get_recognition_results():
    """Get the current recognition results."""
    return recognition_results

def reset_recognition():
    """Reset the recognition system for a new session."""
    global processed_names, recognition_results
    processed_names.clear()
    recognition_results.clear()
    
    # Clear pending records file
    if PENDING_RECORDS_FILE.exists():
        PENDING_RECORDS_FILE.unlink()

def optimize_camera_settings(cap):
    """Optimize camera settings for motion and background operation"""
    try:
        # Set larger buffer size to handle frame drops
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Shorter exposure time (if supported)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        
        # Increase FPS if supported
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enable autofocus if available
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Set resolution to a reasonable size (adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Print current settings for verification
        print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"Buffer Size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    except Exception as e:
        print(f"Warning: Could not optimize camera settings: {e}")

def main():
    """Main function to start face tracking and recognition threads."""
    try:
        file_name = "./face_tracking/config/config_tracking.yaml"
        config_tracking = load_config(file_name)
        
        # Update tracking configuration for better recovery
        config_tracking.update({
            "track_thresh": 0.25,  # Lower threshold to maintain tracking
            "track_buffer": 60,    # Increase buffer to maintain lost tracks longer
            "match_thresh": 0.8,   # Lower threshold for matching
            "min_box_area": 100,   # Lower minimum box area
            "aspect_ratio_thresh": 1.6  # Increase aspect ratio threshold
        })
        
        # Start tracking thread
        thread_track = threading.Thread(
            target=tracking,
            args=(detector, config_tracking,)
        )
        thread_track.start()
        
        # Start recognition thread
        thread_recognize = threading.Thread(target=recognize)
        thread_recognize.start()
        
        # Wait for threads to complete
        thread_track.join()
        thread_recognize.join()
    except KeyboardInterrupt:
        print("Main thread received KeyboardInterrupt. Initiating shutdown...")
        global SHUTDOWN_FLAG
        SHUTDOWN_FLAG = True
        thread_track.join()
        thread_recognize.join()

if __name__ == "__main__":
    main()


