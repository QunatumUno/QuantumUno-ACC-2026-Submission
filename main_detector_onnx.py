import os
import signal
import time
from threading import Thread, Lock
import cv2
import numpy as np
import keyboard
from pal.products.qcar import QCar, QCarCameras

# for onnx
from qcar_detector_onnx import load_model, infer_on_frame

# ================= Configuration =================
CONTROLLER_UPDATE_RATE = 100
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410

# ================= State Management =================
class State:
    def __init__(self):
        self.kill = False
        # Driving Control
        self.throttle = 0.0
        self.steering = 0.0
        # Threading / Vision Variables
        self.lock = Lock()
        self.raw_frame = None
        self.latest_detections = []
        self.frame_id = 0  # Tracks the latest captured frame

state = State()

# ================= Helper Functions =================
def sig_handler(*args): 
    state.kill = True
    
signal.signal(signal.SIGINT, sig_handler)

def handle_arrow(key):
    t_step = 0.01
    s_step = 0.1
    
    if key == 'up': 
        state.throttle = np.clip(state.throttle + t_step, -0.3, 0.3)
    elif key == 'down': 
        state.throttle = np.clip(state.throttle - t_step, -0.3, 0.3)
    elif key == 'left': 
        state.steering = np.clip(state.steering - s_step, -0.6, 0.6)
    elif key == 'right': 
        state.steering = np.clip(state.steering + s_step, -0.6, 0.6)

# ================= Background Vision Thread =================
def yolo_worker(yolo_model):
    """
    Continuously runs YOLOv5 in the background.
    Always skips to the latest frame to minimize detection lag.
    """
    print(">> Vision Thread Started.")
    last_processed_id = -1

    while not state.kill:
        # Grab the latest frame only if it's newer than what we last processed
        with state.lock:
            current_id = state.frame_id
            frame_to_process = (
                state.raw_frame.copy()
                if state.raw_frame is not None and current_id != last_processed_id
                else None
            )

        if frame_to_process is not None:
            last_processed_id = current_id
            # Run inference outside the lock so we don't block the main loop
            _, dets = infer_on_frame(yolo_model, frame_to_process)
            with state.lock:
                state.latest_detections = dets
        else:
            time.sleep(0.005)  # Short sleep to avoid busy-waiting

# ================= Main Hardware Loop =================
def controlLoop():
    print(">> Initializing QCar Hardware...")
    cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
    cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True)
    
    with qcar, cameras:
        print(">> Main Control Loop Running...")
        while not state.kill:
            # 1. Read Hardware
            qcar.read()
            if cameras.csiFront.read():
                with state.lock:
                    state.raw_frame = cameras.csiFront.imageData.copy()
                    state.frame_id += 1  # Increment so vision thread knows a new frame is ready
            
            # 2. Fetch Latest Raw Frame and Detections
            with state.lock:
                display_img = state.raw_frame.copy() if state.raw_frame is not None else None
                current_dets = list(state.latest_detections)

            # 3. Draw UI Overlays
            if display_img is not None:
                
                # Draw Bounding Boxes from the latest detections
                for det in current_dets:
                    x1, y1, x2, y2, conf, cls_id, label = det
                    
                    # Draw a bright blue box
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw label text slightly above the box
                    label_text = f"{label} {conf:.2f}"
                    text_y = y1 - 10 if y1 > 20 else y1 + 20
                    cv2.putText(display_img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # UI at the bottom left
                WHITE = (255, 255, 255)
                bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}"
                cv2.putText(display_img, bot_str, (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
                
                det_str = f"Objects Detected: {len(current_dets)}"
                cv2.putText(display_img, det_str, (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("QCar View", display_img)
                cv2.waitKey(1)
            
            # 4. Write Hardware Commands
            qcar.write(state.throttle, state.steering)
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Setup Keyboard callbacks
    keyboard.add_hotkey('up', lambda: handle_arrow('up'))
    keyboard.add_hotkey('down', lambda: handle_arrow('down'))
    keyboard.add_hotkey('left', lambda: handle_arrow('left'))
    keyboard.add_hotkey('right', lambda: handle_arrow('right'))
    keyboard.add_hotkey('q', lambda: setattr(state, 'kill', True))

    print("--- STARTING QCAR SYSTEM ---")
    
    # 1. Load the Model first
    print(">> Loading YOLOv5 Model. Please wait...")
    model = load_model()
    
    if model is None:
        print("[FATAL] Could not load model. Exiting.")
        exit(1)
        
    # Warmup the model with a blank tensor so the first frame isn't slow
    print(">> Warming up model...")
    dummy_img = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
    infer_on_frame(model, dummy_img)
    
    # 2. Start the Vision Thread
    vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
    vision_thread.start()
    
    # 3. Start the Hardware Control Thread
    control_thread = Thread(target=controlLoop)
    control_thread.start()
    
    # Keep main execution alive until 'q' is pressed
    while control_thread.is_alive() and not state.kill:
        time.sleep(1)
        
    state.kill = True
    print("--- SYSTEM SHUTDOWN COMPLETE ---")