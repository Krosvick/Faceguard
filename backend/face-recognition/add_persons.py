import argparse
import os
import shutil
import os.path as osp
import warnings

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Ignore the specific FutureWarning from PyTorch
warnings.filterwarnings('ignore', category=FutureWarning, 
                       message='`torch.cuda.amp.autocast.*')

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute paths to the model files
current_dir = os.path.dirname(os.path.abspath(__file__))
scrfd_model_path = os.path.join(current_dir, "face_detection", "scrfd", "weights", "scrfd_2.5g_bnkps.onnx")
arcface_model_path = os.path.join(current_dir, "face_recognition", "arcface", "weights", "arcface_r100.pth")

print(f"Using device: {device}")
print(f"SCRFD model path: {scrfd_model_path}")
print(f"ArcFace model path: {arcface_model_path}")

# Initialize the face detector
detector = SCRFD(model_file=scrfd_model_path)

# Initialize the face recognizer with absolute path
recognizer = iresnet_inference(
    model_name="r100", 
    path=arcface_model_path, 
    device=device
)

@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    try:
        # Define a series of image preprocessing steps
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert the image to RGB format
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Apply the defined preprocessing to the image
        face_image = face_preprocess(face_image).unsqueeze(0).to(device)

        # Use the model to obtain facial features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emb_img_face = recognizer(face_image)[0].cpu().numpy()

        # Normalize the features
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)
        return images_emb
    except Exception as e:
        print(f"Error processing face image: {e}")
        return None

def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """
    Add a new person to the face recognition database.
    """
    print(f"\nProcessing images from: {add_persons_dir}")
    
    # Initialize lists to store names and features of added images
    images_name = []
    images_emb = []

    try:
        # Read the folder with images of the new person, extract faces, and save them
        for name_person in os.listdir(add_persons_dir):
            person_image_path = os.path.join(add_persons_dir, name_person)
            
            if not os.path.isdir(person_image_path):
                continue

            print(f"\nProcessing person: {name_person}")

            # Create a directory to save the faces of the person
            person_face_path = os.path.join(faces_save_dir, name_person)
            os.makedirs(person_face_path, exist_ok=True)

            for image_name in os.listdir(person_image_path):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                print(f"Processing image: {image_name}")
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                if input_image is None:
                    print(f"Could not read image: {image_name}")
                    continue

                # Detect faces and landmarks using the face detector
                bboxes, landmarks = detector.detect(image=input_image)

                if len(bboxes) == 0:
                    print(f"No faces detected in: {image_name}")
                    continue

                # Extract faces
                for i in range(len(bboxes)):
                    # Get the number of files in the person's path
                    number_files = len(os.listdir(person_face_path))

                    # Get the location of the face
                    x1, y1, x2, y2, score = bboxes[i]

                    # Extract the face from the image
                    face_image = input_image[int(y1):int(y2), int(x1):int(x2)]

                    # Extract features from the face
                    features = get_feature(face_image=face_image)
                    if features is None:
                        print(f"Could not extract features from face in: {image_name}")
                        continue

                    # Save the face and add features
                    face_filename = f"{number_files}.jpg"
                    face_path = os.path.join(person_face_path, face_filename)
                    cv2.imwrite(face_path, face_image)
                    
                    images_emb.append(features)
                    images_name.append(name_person)
                    print(f"Saved face: {face_filename}")

        # Check if no new person is found
        if not images_emb:
            print("No new faces processed!")
            return None

        # Convert lists to arrays
        images_emb = np.array(images_emb)
        images_name = np.array(images_name)

        # Read existing features if available
        features = read_features(features_path)

        if features is not None:
            # Unpack existing features
            old_images_name, old_images_emb = features

            # Combine new features with existing features
            images_name = np.hstack((old_images_name, images_name))
            images_emb = np.vstack((old_images_emb, images_emb))
            print("Updated existing features!")
        else:
            print("Created new features file!")

        # Save the combined features
        np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

        # Move the data of the new person to the backup data directory
        for sub_dir in os.listdir(add_persons_dir):
            dir_to_move = os.path.join(add_persons_dir, sub_dir)
            if os.path.isdir(dir_to_move):
                backup_path = os.path.join(backup_dir, sub_dir)
                shutil.move(dir_to_move, backup_path)
                print(f"Moved {sub_dir} to backup")

        print("Successfully processed all faces!")

    except Exception as e:
        print(f"Error processing faces: {e}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
