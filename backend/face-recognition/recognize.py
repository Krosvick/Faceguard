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
from typing import Dict, Optional
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

# Device configuration



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# Face detector (choose one)



detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")







# Face recognizer



recognizer = iresnet_inference(



    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device



)







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



tracking_queue = queue.Queue(maxsize=10)  # Adjust maxsize as needed

ROOM_ID = "LC6"
DISPLAY_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for display purposes
RECORD_CONFIDENCE_THRESHOLD = 0.6    # Higher threshold for recording attendance
RECOGNITION_COOLDOWN = 5  # Seconds between recognitions
last_recognition_time: Dict[str, datetime] = {}
best_recognition: Dict[str, Optional[dict]] = {}
processed_names = set()  # Track which names have been processed
recognition_results = {}  # Store recognition results

RECORDS_DIR = Path("./attendance_records")
PENDING_RECORDS_FILE = RECORDS_DIR / "pending_records.json"
BATCH_SIZE = 10  # Number of records to store before writing to file

# Initialize records directory
RECORDS_DIR.mkdir(exist_ok=True)

API_BASE_URL = "http://localhost:3000/api"  # Change this to your Next.js API URL
SEND_INTERVAL = 5  # Seconds between sending batches to API
MAX_RETRIES = 3

# Add new variable to track class end times
class_end_times = {}

def load_config(file_name):



    """



    Load a YAML configuration file.







    Args:



        file_name (str): The path to the YAML configuration file.







    Returns:



        dict: The loaded configuration as a dictionary.



    """



    with open(file_name, "r") as stream:



        try:



            return yaml.safe_load(stream)



        except yaml.YAMLError as exc:



            print(exc)

def process_tracking(frame, detector, tracker, args, frame_id, fps):



    """



    Process tracking for a frame.







    Args:



        frame: The input frame.



        detector: The face detector.



        tracker: The object tracker.



        args (dict): Tracking configuration parameters.



        frame_id (int): The frame ID.



        fps (float): Frames per second.







    Returns:



        numpy.ndarray: The processed tracking image.



    """



    # Face detection and tracking



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







    # Update tracker



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



        tracking_queue.put(tracking_data, timeout=1)



    except queue.Full:



        print("Tracking queue is full. Skipping frame.")







    return tracking_image

@torch.no_grad()



def get_feature(face_image):



    """



    Extract features from a face image.







    Args:



        face_image: The input face image.







    Returns:



        numpy.ndarray: The extracted features.



    """



    face_preprocess = transforms.Compose(



        [



            transforms.ToTensor(),



            transforms.Resize((112, 112)),



            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),



        ]



    )







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



    """



    Recognize a face image.







    Args:



        face_image: The input face image.







    Returns:



        tuple: A tuple containing the recognition score and name.



    """



    # Get feature from face



    query_emb = get_feature(face_image)







    score, id_min = compare_encodings(query_emb, images_embs)

    name = images_names[id_min]



    score = score[0]







    return score, name

def mapping_bbox(box1, box2):



    """



    Calculate the Intersection over Union (IoU) between two bounding boxes.







    Args:



        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).



        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).







    Returns:



        float: The IoU score.



    """



    # Calculate the intersection area



    x_min_inter = max(box1[0], box2[0])



    y_min_inter = max(box1[1], box2[1])



    x_max_inter = min(box1[2], box2[2])



    y_max_inter = min(box1[3], box2[3])







    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(



        0, y_max_inter - y_min_inter + 1



    )







    # Calculate the area of each bounding box



    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)



    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)







    # Calculate the union area



    union_area = area_box1 + area_box2 - intersection_area







    # Calculate IoU



    iou = intersection_area / union_area







    return iou

def tracking(detector, args):



    """



    Face tracking in a separate thread.







    Args:



        detector: The face detector.



        args (dict): Tracking configuration parameters.



    """



    # Initialize variables for measuring frame rate



    start_time = time.time_ns()



    frame_count = 0



    fps = -1







    # Initialize a tracker and a timer



    tracker = BYTETracker(args=args, frame_rate=30)



    frame_id = 0







    cap = cv2.VideoCapture(0)







    while True:



        ret, img = cap.read()



        if not ret:



            print("Failed to grab frame")



            break







        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)







        # Calculate and display the frame rate



        frame_count += 1



        if frame_count >= 30:



            end_time = time.time_ns()



            fps = 1e9 * frame_count / (end_time - start_time)



            frame_count = 0



            start_time = time.time_ns()







        cv2.imshow("Face Recognition", tracking_image)







        # Check for user exit input



        ch = cv2.waitKey(1)



        if ch == 27 or ch == ord("q") or ch == ord("Q"):



            break







    cap.release()



    cv2.destroyAllWindows()

def save_face_image(face_image, name: str, quality: float):
    """Save the face image with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recognized_faces/{name}_{timestamp}_{quality:.2f}.jpg"
    cv2.imwrite(filename, face_image)
    return filename

def encode_image_base64(face_image):
    """
    Convert CV2 image to base64 string.

    Args:
        face_image (numpy.ndarray): CV2 image in BGR format

    Returns:
        str: Base64 encoded image string
    """
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
    """Save pending records to JSON file."""
    with open(PENDING_RECORDS_FILE, "w") as f:
        json.dump(records, f, indent=2)

def clear_sent_records(successful_records):
    """Remove successfully sent records from pending records."""
    pending_records = load_pending_records()
    remaining_records = [r for r in pending_records 
                        if r["timestamp"] not in [sr["timestamp"] for sr in successful_records]]
    save_pending_records(remaining_records)

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, URLError), max_tries=MAX_RETRIES)
def send_records_to_api(records):
    """
    Send records to Next.js API with retry logic.
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
                # Parse the ISO string and convert to timezone-naive datetime
                end_time = datetime.fromisoformat(data['classEndTime'].replace('Z', '+00:00'))
                room = records[0]['room']
                # Store as timezone-naive datetime
                class_end_times[room] = end_time.replace(tzinfo=None)
            print(f"Successfully sent {len(records)} records to API")
            return records
        else:
            print(f"Failed to send records. Status: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error sending records to API: {e}")
        return []

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

def recognize():



    """Face recognition in a separate thread."""
    global processed_names, recognition_results
    pending_records = load_pending_records()
    batch_count = 0
    last_send_time = time.time()

    while True:



        try:



            tracking_data = tracking_queue.get(timeout=1)



        except queue.Empty:
            # Check if it's time to send pending records
            current_time = time.time()
            if pending_records and (current_time - last_send_time) >= SEND_INTERVAL:
                successful_records = send_records_to_api(pending_records)
                if successful_records:
                    clear_sent_records(successful_records)
                    pending_records = load_pending_records()
                last_send_time = current_time
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

                for score, name, tid, quality, face_img in zip(scores, names, ids_to_update, quality_scores, face_images):
                    if score >= DISPLAY_CONFIDENCE_THRESHOLD:
                        caption = f"{name}:{score:.2f}"
                        
                        if (score >= RECORD_CONFIDENCE_THRESHOLD and 
                            should_record_attendance(name, ROOM_ID, current_time)):
                            
                            face_image_base64 = encode_image_base64(face_img)
                            
                            record = {
                                "name": name,
                                "confidence": float(score),
                                "quality": float(quality),
                                "timestamp": current_time.isoformat(),
                                "room": ROOM_ID,
                                "image": face_image_base64,
                                "status": "pending"
                            }
                            
                            recognition_results[name] = record
                            pending_records.append(record)
                            processed_names.add(name)
                            
                            batch_count += 1
                            if batch_count >= BATCH_SIZE:
                                save_pending_records(pending_records)
                                batch_count = 0
                                
                            # Print the recognition record (excluding the base64 image)
                            print_record = {**record}
                            print_record["image"] = "<<base64_image_data>>"
                            print("New Recognition:", json.dumps(print_record, indent=2))
                    else:
                        caption = "UNKNOWN"
                    
                    id_face_mapping[tid] = caption







            if not tracking_bboxes:



                print("Waiting for a person...")

            # After processing new recognitions, check if it's time to send
            current_time = time.time()
            if pending_records and (current_time - last_send_time) >= SEND_INTERVAL:
                successful_records = send_records_to_api(pending_records)
                if successful_records:
                    clear_sent_records(successful_records)
                    pending_records = load_pending_records()
                last_send_time = current_time

        except Exception as e:



            print(f"Error in recognition thread: {e}")
            # Save any pending records on error
            if pending_records:
                save_pending_records(pending_records)

@torch.no_grad()



def get_feature_batch(face_images):



    """



    Extract features from a batch of face images.







    Args:



        face_images (list): List of face images.







    Returns:



        numpy.ndarray: Extracted features.



    """



    face_preprocess = transforms.Compose(



        [



            transforms.ToTensor(),



            transforms.Resize((112, 112)),



            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),



        ]



    )







    # Convert to RGB and preprocess



    processed_faces = [face_preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in face_images]



    processed_faces = torch.stack(processed_faces).to(device)







    # Inference to get features



    emb_imgs = recognizer(processed_faces).cpu().numpy()







    # Normalize



    images_embs = emb_imgs / np.linalg.norm(emb_imgs, axis=1, keepdims=True)







    return images_embs

def compare_encodings_batch(features, reference_features):



    """



    Compare batch of features with reference features.







    Args:



        features (numpy.ndarray): Batch of query features.



        reference_features (numpy.ndarray): Reference features.







    Returns:



        tuple: Scores and corresponding reference indices.



    """



    # Compute cosine similarity



    similarities = np.dot(features, reference_features.T)



    scores = similarities.max(axis=1)



    ids_min = similarities.argmax(axis=1)







    return scores, [images_names[idx] for idx in ids_min]

def check_face_quality(face_image):



    """



    Check the quality of a face image.

    Args:



        face_image (numpy.ndarray): Input face image

    Returns:



        bool: True if face quality is acceptable, False otherwise



        float: Quality score



    """



    # Calculate basic quality metrics



    



    # 1. Check image brightness



    brightness = np.mean(face_image)



    



    # 2. Check image contrast



    contrast = np.std(face_image)



    



    # 3. Check face size (assuming minimum 112x112 for ArcFace)



    height, width = face_image.shape[:2]



    min_size = min(height, width)



    



    # 4. Check blur using Laplacian variance



    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)



    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()



    



    # Define quality thresholds



    quality_score = 0.0



    quality_score += 0.3 * (1 if brightness > 40 and brightness < 250 else 0)  # Check if not too dark or bright



    quality_score += 0.3 * (1 if contrast > 20 else 0)  # Check if enough contrast



    quality_score += 0.2 * (1 if min_size >= 112 else 0)  # Check minimum size



    quality_score += 0.2 * (1 if blur_score > 100 else 0)  # Check if not too blurry



    



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

def main():



    """Main function to start face tracking and recognition threads."""



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



        args=(



            detector,



            config_tracking,



        ),



    )



    thread_track.start()







    # Start recognition thread



    thread_recognize = threading.Thread(target=recognize)



    thread_recognize.start()







    # Join threads to ensure they complete



    thread_track.join()



    thread_recognize.join()

if __name__ == "__main__":



    main()


