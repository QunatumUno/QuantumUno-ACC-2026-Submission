import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Setup paths to ensure YOLOv5 internal imports work
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import YOLOv5 utilities
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# --- UPDATED: Your specific classes ---
class_names = [
    "cone", "green_light", "pedestrian", "red_light", 
    "roundabout", "stop", "yellow_light", "yield"
]

def load_model(
    # --- UPDATED: Your specific weights path ---
    # Note: YOLOv5 usually puts the weights inside a 'weights' subfolder. 
    # If your best.pt is directly inside yolov5s_1, remove '/weights' from this string.
    weights="runs/acc/yolov5s_1/weights/best.pt",
    device='' # Leave empty to auto-select GPU if available, else CPU
):
    device = select_device(device)
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    print(f">> YOLOv5 Model loaded from {weights} on {device}")
    return model

def infer_on_frame(
    model,
    im0,
    imgsz=(832, 832), # --- UPDATED: Match your training --img size ---
    conf_thres=0.7,
    iou_thres=0.45
):
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Preprocess (Resizes 820x410 to 832x832 with padding automatically)
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # Inference
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    det = pred[0] if pred else torch.tensor([])

    # Draw bounding boxes
    annotator = Annotator(im0.copy(), line_width=2, example=str(class_names))
    
    # Store clean detection data to return to the QCar script
    # Format: [ [x1, y1, x2, y2, confidence, class_index, class_name], ... ]
    detections_list = []

    if len(det):
        # Rescale boxes back to original 820x410 image size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        
        for *xyxy, conf, cls in det:
            cls_id = int(cls)
            conf_val = float(conf)
            label_name = class_names[cls_id]
            
            # Draw on image
            label = f"{label_name} {conf_val:.2f}"
            annotator.box_label(xyxy, label, color=colors(cls_id, True))
            
            # Save raw data for QCar logic
            detections_list.append([
                int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), 
                conf_val, cls_id, label_name
            ])

    annotated_im0 = annotator.result()

    return annotated_im0, detections_list