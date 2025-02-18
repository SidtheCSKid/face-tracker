import os
import argparse

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='detector.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.7)



def process_image(img, track, annotate):
    # Define the margin factor (start small, then smoothly adjust)
    target_margin = 1.2
    smooth_margin = 1.0  # Start with no margin

    alpha = 0.1  # Smoothing factor (higher = faster change)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    with FaceDetector.create_from_options(options) as detector:
        face_detector_result = detector.detect(mp_image)
        if face_detector_result.detections:
            # iterate through detections and calculate area of bounding box and get detection with largest area
            primary_detection = max(face_detector_result.detections, key=lambda detection: detection.bounding_box.width * detection.bounding_box.height)
                
            bbox = primary_detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
            center_x_bbox = bbox.origin_x + bbox.width/2
            center_y_bbox = bbox.origin_y + bbox.height/2

            if annotate:
                # Draw center point
                cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
                cv2.circle(img, (int(center_x_bbox), int(center_y_bbox)), 5, (0, 0, 255), -1)
                # Draw keypoints
                for keypoint in primary_detection.keypoints:
                    x = int(keypoint.x * img.shape[1])  # Convert normalized x to pixel
                    y = int(keypoint.y * img.shape[0])  # Convert normalized y to pixel

                    # Draw the keypoint
                    cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)  # Green dots

            if track:
                # Smoothly adjust the zoom level
                smooth_margin += alpha * (target_margin - smooth_margin)

                expanded_width = bbox.width * smooth_margin
                expanded_height = bbox.height * smooth_margin

                scale_factor = max(img.shape[1] / expanded_width, img.shape[0] / expanded_height) * 0.3

                margin_factor = 1.2
                
                # Resize the image
                new_width = int(img.shape[1] * scale_factor) 
                new_height = int(img.shape[0] * scale_factor) 
                resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Calculate new bounding box center in the resized image
                center_x = int(center_x_bbox * scale_factor)
                center_y = int(center_y_bbox * scale_factor)

                # Define the crop region centered around the bounding box
                crop_x1 = max(center_x - img.shape[1] // 2, 0) 
                crop_y1 = max(center_y - img.shape[0] // 2, 0) 
                crop_x2 = min(crop_x1 + img.shape[1], resized_img.shape[1]) 
                crop_y2 = min(crop_y1 + img.shape[0], resized_img.shape[0]) 

                # Crop the image
                cropped_img = resized_img[crop_y1:crop_y2, crop_x1:crop_x2]

                resized_img = cv2.resize(cropped_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                img = resized_img


    return img

track = False
annotate = False

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    img = process_image(frame, track, annotate)
    cv2.imshow("image", img)

    # break if key pressed is q
    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):  # If 'q' is pressed
        break
    elif key == ord('t'):  # If 't' is pressed
        track = not track  # Toggle track
    elif key == ord('s'):
        annotate = not annotate

cap.release()