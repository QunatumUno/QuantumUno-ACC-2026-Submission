# import os
# import signal
# import time
# from threading import Thread, Lock
# import cv2
# import numpy as np
# import keyboard
# from pal.products.qcar import QCar, QCarCameras

# # Import from our custom module for pt
# # from qcar_detector import load_model, infer_on_frame
# # for onnx
# from qcar_detector import load_model, infer_on_frame

# # ================= Configuration =================
# CONTROLLER_UPDATE_RATE = 100
# CSI_WIDTH, CSI_HEIGHT = 820, 410
# DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410

# # ================= State Management =================
# class State:
#     def __init__(self):
#         self.kill = False
#         # Driving Control
#         self.throttle = 0.0
#         self.steering = 0.0
#         # Threading / Vision Variables
#         self.lock = Lock()
#         self.raw_frame = None
#         self.latest_detections = []

# state = State()

# # ================= Helper Functions =================
# def sig_handler(*args): 
#     state.kill = True
    
# signal.signal(signal.SIGINT, sig_handler)

# def handle_arrow(key):
#     t_step = 0.01
#     s_step = 0.1
    
#     if key == 'up': 
#         state.throttle = np.clip(state.throttle + t_step, -0.3, 0.3)
#     elif key == 'down': 
#         state.throttle = np.clip(state.throttle - t_step, -0.3, 0.3)
#     elif key == 'left': 
#         state.steering = np.clip(state.steering - s_step, -0.6, 0.6)
#     elif key == 'right': 
#         state.steering = np.clip(state.steering + s_step, -0.6, 0.6)

# # ================= Background Vision Thread =================
# def yolo_worker(yolo_model):
#     """
#     Continuously runs YOLOv5 in the background.
#     """
#     print(">> Vision Thread Started.")
#     while not state.kill:
#         # Safely grab a copy of the latest raw frame
#         with state.lock:
#             frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None
            
#         if frame_to_process is not None:
#             # Run heavy inference (Outside the lock to prevent blocking)
#             # We ignore the annotated image and only grab the raw detection data
#             _, dets = infer_on_frame(yolo_model, frame_to_process)
            
#             # Safely update the bounding box data back to the main state
#             with state.lock:
#                 state.latest_detections = dets
#         else:
#             time.sleep(0.01) # Wait for camera to boot up

# # ================= Main Hardware Loop =================
# def controlLoop():
#     print(">> Initializing QCar Hardware...")
#     cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
#     qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
#     cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True)
    
#     with qcar, cameras:
#         print(">> Main Control Loop Running...")
#         while not state.kill:
#             # 1. Read Hardware
#             qcar.read()
#             if cameras.csiFront.read():
#                 with state.lock:
#                     state.raw_frame = cameras.csiFront.imageData.copy()
            
#             # 2. Fetch Fast Camera Frame and Latest Detections
#             with state.lock:
#                 # FIX: We now display the RAW frame for buttery smooth 30 FPS video
#                 display_img = state.raw_frame.copy() if state.raw_frame is not None else None
#                 current_dets = list(state.latest_detections)

#             # 3. Draw UI Overlays
#             if display_img is not None:
                
#                 # FIX: Draw Bounding Boxes dynamically in the fast thread
#                 for det in current_dets:
#                     # Unpack the detection data from qcar_detector.py
#                     x1, y1, x2, y2, conf, cls_id, label = det
                    
#                     # Draw a bright blue box
#                     cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
#                     # Draw label text slightly above the box
#                     label_text = f"{label} {conf:.2f}"
#                     text_y = y1 - 10 if y1 > 20 else y1 + 20 # Prevent text from going off screen
#                     cv2.putText(display_img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#                 # FIX: Moved UI to the BOTTOM LEFT of the screen to prevent overlaps
#                 WHITE = (255, 255, 255)
#                 bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}"
#                 cv2.putText(display_img, bot_str, (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
                
#                 det_str = f"Objects Detected: {len(current_dets)}"
#                 cv2.putText(display_img, det_str, (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#                 cv2.imshow("QCar View", display_img)
#                 cv2.waitKey(1)
            
#             # 4. Write Hardware Commands
#             qcar.write(state.throttle, state.steering)
            
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # Setup Keyboard callbacks
#     keyboard.add_hotkey('up', lambda: handle_arrow('up'))
#     keyboard.add_hotkey('down', lambda: handle_arrow('down'))
#     keyboard.add_hotkey('left', lambda: handle_arrow('left'))
#     keyboard.add_hotkey('right', lambda: handle_arrow('right'))
#     keyboard.add_hotkey('q', lambda: setattr(state, 'kill', True))

#     print("--- STARTING QCAR SYSTEM ---")
    
#     # 1. Load the Model first
#     print(">> Loading YOLOv5 Model. Please wait...")
#     model = load_model()
    
#     if model is None:
#         print("[FATAL] Could not load model. Exiting.")
#         exit(1)
        
#     # Warmup the model with a blank tensor so the first frame isn't slow
#     print(">> Warming up model...")
#     dummy_img = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
#     infer_on_frame(model, dummy_img)
    
#     # 2. Start the Vision Thread
#     vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
#     vision_thread.start()
    
#     # 3. Start the Hardware Control Thread
#     control_thread = Thread(target=controlLoop)
#     control_thread.start()
    
#     # Keep main execution alive until 'q' is pressed
#     while control_thread.is_alive() and not state.kill:
#         time.sleep(1)
        
#     state.kill = True
#     print("--- SYSTEM SHUTDOWN COMPLETE ---")

# import os
# import signal
# import time
# from threading import Thread, Lock
# import cv2
# import numpy as np
# import pathlib
# from pal.products.qcar import QCar, QCarCameras

# # Import from our custom module for pt
# # from qcar_detector import load_model, infer_on_frame
# # for onnx
# from qcar_detector import load_model, infer_on_frame

# # ================= Configuration =================
# CONTROLLER_UPDATE_RATE = 100
# CSI_WIDTH, CSI_HEIGHT = 820, 410
# DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410

# # ================= State Management =================
# class State:
#     def __init__(self):
#         self.kill = False
#         # Driving Control
#         self.throttle = 0.0
#         self.steering = 0.0
#         # Threading / Vision Variables
#         self.lock = Lock()
#         self.raw_frame = None
#         self.latest_detections = []

# state = State()

# # ================= Helper Functions =================
# def sig_handler(*args): 
#     state.kill = True
    
# signal.signal(signal.SIGINT, sig_handler)

# def handle_arrow(key):
#     t_step = 0.01
#     s_step = 0.1
    
#     if key == 'up': 
#         state.throttle = np.clip(state.throttle + t_step, -0.3, 0.3)
#     elif key == 'down': 
#         state.throttle = np.clip(state.throttle - t_step, -0.3, 0.3)
#     elif key == 'left': 
#         state.steering = np.clip(state.steering - s_step, -0.6, 0.6)
#     elif key == 'right': 
#         state.steering = np.clip(state.steering + s_step, -0.6, 0.6)

# def handle_keypress(k):
#     if k in (ord('q'), ord('Q')):
#         state.kill = True
#         return

#     if k in (2490368, 82, 65362):
#         handle_arrow('up')
#     elif k in (2621440, 84, 65364):
#         handle_arrow('down')
#     elif k in (2424832, 81, 65361):
#         handle_arrow('left')
#     elif k in (2555904, 83, 65363):
#         handle_arrow('right')

# # ================= Background Vision Thread =================
# def yolo_worker(yolo_model):
#     """
#     Continuously runs YOLOv5 in the background.
#     """
#     print(">> Vision Thread Started.")
#     while not state.kill:
#         # Safely grab a copy of the latest raw frame
#         with state.lock:
#             frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None
            
#         if frame_to_process is not None:
#             # Run heavy inference (Outside the lock to prevent blocking)
#             # We ignore the annotated image and only grab the raw detection data
#             _, dets = infer_on_frame(yolo_model, frame_to_process)
            
#             # Safely update the bounding box data back to the main state
#             with state.lock:
#                 state.latest_detections = dets
#         else:
#             time.sleep(0.01) # Wait for camera to boot up

# # ================= Main Hardware Loop =================
# def controlLoop():
#     print(">> Initializing QCar Hardware...")
#     cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
#     qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
#     cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True)
    
#     with qcar, cameras:
#         print(">> Main Control Loop Running...")
#         while not state.kill:
#             # 1. Read Hardware
#             qcar.read()
#             if cameras.csiFront.read():
#                 with state.lock:
#                     state.raw_frame = cameras.csiFront.imageData.copy()
            
#             # 2. Fetch Fast Camera Frame and Latest Detections
#             with state.lock:
#                 # FIX: We now display the RAW frame for buttery smooth 30 FPS video
#                 display_img = state.raw_frame.copy() if state.raw_frame is not None else None
#                 current_dets = list(state.latest_detections)

#             # 3. Draw UI Overlays
#             if display_img is not None:
                
#                 # FIX: Draw Bounding Boxes dynamically in the fast thread
#                 for det in current_dets:
#                     # Unpack the detection data from qcar_detector.py
#                     x1, y1, x2, y2, conf, cls_id, label = det
                    
#                     # Draw a bright blue box
#                     cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
#                     # Draw label text slightly above the box
#                     label_text = f"{label} {conf:.2f}"
#                     text_y = y1 - 10 if y1 > 20 else y1 + 20 # Prevent text from going off screen
#                     cv2.putText(display_img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#                 # FIX: Moved UI to the BOTTOM LEFT of the screen to prevent overlaps
#                 WHITE = (255, 255, 255)
#                 bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}"
#                 cv2.putText(display_img, bot_str, (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
                
#                 det_str = f"Objects Detected: {len(current_dets)}"
#                 cv2.putText(display_img, det_str, (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#                 cv2.imshow("QCar View", display_img)
#                 k = cv2.waitKeyEx(1)
#                 if k != -1:
#                     handle_keypress(k)
            
#             # 4. Write Hardware Commands
#             qcar.write(state.throttle, state.steering)
            
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("--- STARTING QCAR SYSTEM ---")
    
    # pathlib.WindowsPath = pathlib.PosixPath
#     # 1. Load the Model first
#     print(">> Loading YOLOv5 Model. Please wait...")
#     model = load_model()
    
#     if model is None:
#         print("[FATAL] Could not load model. Exiting.")
#         exit(1)
        
#     # Warmup the model with a blank tensor so the first frame isn't slow
#     print(">> Warming up model...")
#     dummy_img = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
#     infer_on_frame(model, dummy_img)
    
#     # 2. Start the Vision Thread
#     vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
#     vision_thread.start()
    
#     # 3. Start the Hardware Control Thread
#     control_thread = Thread(target=controlLoop)
#     control_thread.start()
    
#     control_thread.join()
#     state.kill = True
#     print("--- SYSTEM SHUTDOWN COMPLETE ---")


# import os
# import signal
# import time
# from threading import Thread, Lock
# import cv2
# import numpy as np
# import pathlib
# from pal.products.qcar import QCar, QCarCameras

# # Import from our custom module for pt
# # from qcar_detector import load_model, infer_on_frame
# # for onnx
# from qcar_detector import load_model, infer_on_frame

# # ================= Configuration =================
# CONTROLLER_UPDATE_RATE = 100
# CSI_WIDTH, CSI_HEIGHT = 820, 410
# DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410

# # ================= State Management =================
# class State:
#     def __init__(self):
#         self.kill = False
#         # Driving Control
#         self.throttle = 0.0
#         self.steering = 0.0
#         # Threading / Vision Variables
#         self.lock = Lock()
#         self.raw_frame = None
#         self.latest_detections = []

# state = State()

# # ================= Helper Functions =================
# def sig_handler(*args): 
#     state.kill = True
    
# signal.signal(signal.SIGINT, sig_handler)

# def handle_arrow(key):
#     t_step = 0.01
#     s_step = 0.1
    
#     if key == 'up': 
#         state.throttle = np.clip(state.throttle + t_step, -0.3, 0.3)
#     elif key == 'down': 
#         state.throttle = np.clip(state.throttle - t_step, -0.3, 0.3)
#     elif key == 'left': 
#         state.steering = np.clip(state.steering + s_step, -0.6, 0.6)
#     elif key == 'right': 
#         state.steering = np.clip(state.steering - s_step, -0.6, 0.6)

# def handle_keypress(k):
#     if k is None or k == -1:
#         return

#     if k in (ord('q'), ord('Q'), 27):
#         state.kill = True
#         return

#     UP_CODES    = {2490368, 82, 65362}
#     DOWN_CODES  = {2621440, 84, 65364}
#     LEFT_CODES  = {2424832, 81, 65361}
#     RIGHT_CODES = {2555904, 83, 65363}

#     if k in UP_CODES:
#         handle_arrow('up')
#     elif k in DOWN_CODES:
#         handle_arrow('down')
#     elif k in LEFT_CODES:
#         handle_arrow('left')
#     elif k in RIGHT_CODES:
#         handle_arrow('right')

# # ================= Background Vision Thread =================
# def yolo_worker(yolo_model):
#     """
#     Continuously runs YOLOv5 in the background.
#     """
#     print(">> Vision Thread Started.")
#     while not state.kill:
#         # Safely grab a copy of the latest raw frame
#         with state.lock:
#             frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None
            
#         if frame_to_process is not None:
#             # Run heavy inference (Outside the lock to prevent blocking)
#             # We ignore the annotated image and only grab the raw detection data
#             _, dets = infer_on_frame(yolo_model, frame_to_process)
            
#             # Safely update the bounding box data back to the main state
#             with state.lock:
#                 state.latest_detections = dets
#         else:
#             time.sleep(0.01) # Wait for camera to boot up

# # ================= Main Hardware Loop =================
# def controlLoop():
#     print(">> Initializing QCar Hardware...")
#     cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
#     qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
#     cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True,)
    
#     with qcar, cameras:
#         print(">> Main Control Loop Running...")
#         while not state.kill:
#             # front_ok = cameras.csiFront.read()
#             # left_ok  = cameras.csiLeft.read()
#             # right_ok = cameras.csiRight.read()

#             # print("front", front_ok, "left", left_ok, "right", right_ok)

#             # if front_ok:
#             #     img = cameras.csiFront.imageData
#             # elif left_ok:
#             #     img = cameras.csiLeft.imageData
#             # elif right_ok:
#             #     img = cameras.csiRight.imageData
#             # else:
#             #     img = None
#             # 1. Read Hardware
#             qcar.read()
#             if cameras.csiFront.read():
#                 with state.lock:
#                     state.raw_frame = cameras.csiFront.imageData.copy()
            
#             # 2. Fetch Fast Camera Frame and Latest Detections
#             with state.lock:
#                 display_img = state.raw_frame.copy() if state.raw_frame is not None else None
#                 current_dets = list(state.latest_detections)

#             # Always show something so key events keep working
#             if display_img is None:
#                 display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

#             # 3. Draw UI Overlays
#             if current_dets:
#                 for det in current_dets:
#                     x1, y1, x2, y2, conf, cls_id, label = det
#                     cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     label_text = f"{label} {conf:.2f}"
#                     text_y = y1 - 10 if y1 > 20 else y1 + 20
#                     cv2.putText(display_img, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#             WHITE = (255, 255, 255)
#             bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}"
#             cv2.putText(display_img, bot_str, (20, DISPLAY_HEIGHT - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
#             det_str = f"Objects Detected: {len(current_dets)}"
#             cv2.putText(display_img, det_str, (20, DISPLAY_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#             cv2.imshow("QCar View", display_img)

#             # IMPORTANT: 10ms makes arrow key detection much more reliable
#             k = cv2.waitKeyEx(10)
#             handle_keypress(k)
            
#             # 4. Write Hardware Commands
#             qcar.write(state.throttle, state.steering)
            
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("--- STARTING QCAR SYSTEM ---")
#     pathlib.WindowsPath = pathlib.PosixPath
#     # 1. Load the Model first
#     print(">> Loading YOLOv5 Model. Please wait...")
#     model = load_model()
    
#     if model is None:
#         print("[FATAL] Could not load model. Exiting.")
#         exit(1)
        
#     # Warmup the model with a blank tensor so the first frame isn't slow
#     print(">> Warming up model...")
#     dummy_img = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
#     infer_on_frame(model, dummy_img)
    
#     # 2. Start the Vision Thread
#     vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
#     vision_thread.start()
    
#     # 3. Start the Hardware Control Thread
#     control_thread = Thread(target=controlLoop)
#     control_thread.start()
    
#     control_thread.join()
#     state.kill = True
#     print("--- SYSTEM SHUTDOWN COMPLETE ---")


import os
import signal
import time
from threading import Thread, Lock
import cv2
import numpy as np
import pathlib
from pal.products.qcar import QCar, QCarCameras

from qcar_detector import load_model, infer_on_frame

# ================= Configuration =================
CONTROLLER_UPDATE_RATE = 100
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410

# Choose which stream you want to display (change this)
PREFERRED_CAM = "LEFT"   # "FRONT" or "LEFT" or "RIGHT"

# If preferred cam fails for this many seconds, try fallback cams
FAILOVER_AFTER_SEC = 1.0

# ================= State Management =================
class State:
    def __init__(self):
        self.kill = False
        self.throttle = 0.0
        self.steering = 0.0
        self.lock = Lock()
        self.raw_frame = None
        self.latest_detections = []
        self.cam_name = "NONE"

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
        state.steering = np.clip(state.steering + s_step, -0.6, 0.6)
    elif key == 'right':
        state.steering = np.clip(state.steering - s_step, -0.6, 0.6)

def handle_keypress(k):
    if k is None or k == -1:
        return

    if k in (ord('q'), ord('Q'), 27):
        state.kill = True
        return

    UP_CODES    = {2490368, 82, 65362}
    DOWN_CODES  = {2621440, 84, 65364}
    LEFT_CODES  = {2424832, 81, 65361}
    RIGHT_CODES = {2555904, 83, 65363}

    if k in UP_CODES:
        handle_arrow('up')
    elif k in DOWN_CODES:
        handle_arrow('down')
    elif k in LEFT_CODES:
        handle_arrow('left')
    elif k in RIGHT_CODES:
        handle_arrow('right')

def get_stream(cameras, name: str):
    name = name.upper()
    if name == "FRONT":
        return getattr(cameras, "csiFront", None)
    if name == "LEFT":
        return getattr(cameras, "csiLeft", None)
    if name == "RIGHT":
        return getattr(cameras, "csiRight", None)
    return None

# ================= Background Vision Thread =================
def yolo_worker(yolo_model):
    print(">> Vision Thread Started.")
    while not state.kill:
        with state.lock:
            frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None

        if frame_to_process is not None:
            _, dets = infer_on_frame(yolo_model, frame_to_process)
            with state.lock:
                state.latest_detections = dets
        else:
            time.sleep(0.01)

# ================= Main Hardware Loop =================
def controlLoop():
    print(">> Initializing QCar Hardware...")
    cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)

    # Enable all (so failover is possible), but we will READ ONLY ONE at a time
    cameras = QCarCameras(
        frameWidth=CSI_WIDTH,
        frameHeight=CSI_HEIGHT,
        frameRate=30,
        enableFront=True,
        enableLeft=True,
        enableRight=True,
    )

    preferred_name = PREFERRED_CAM.upper()
    active_name = preferred_name
    last_good_frame_time = time.time()

    with qcar, cameras:
        print(">> Main Control Loop Running...")
        while not state.kill:
            qcar.read()

            img = None

            # 1) Try active camera ONLY (prevents blinking)
            active_stream = get_stream(cameras, active_name)
            if active_stream is not None and active_stream.read():
                img = active_stream.imageData
                last_good_frame_time = time.time()

            # 2) If active cam has failed for a while, fail over (optional)
            if img is None and (time.time() - last_good_frame_time) > FAILOVER_AFTER_SEC:
                for candidate in ["FRONT", "LEFT", "RIGHT"]:
                    stream = get_stream(cameras, candidate)
                    if stream is not None and stream.read():
                        img = stream.imageData
                        active_name = candidate
                        last_good_frame_time = time.time()
                        break

            if img is not None:
                with state.lock:
                    state.raw_frame = img.copy()
                    state.cam_name = active_name

            with state.lock:
                display_img = state.raw_frame.copy() if state.raw_frame is not None else None
                current_dets = list(state.latest_detections)
                cam_name_disp = state.cam_name

            if display_img is None:
                display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

            # Draw detections
            if current_dets:
                for det in current_dets:
                    x1, y1, x2, y2, conf, cls_id, label = det
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_text = f"{label} {conf:.2f}"
                    text_y = y1 - 10 if y1 > 20 else y1 + 20
                    cv2.putText(display_img, label_text, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # UI overlays
            # cv2.putText(display_img, f"CAM: {cam_name_disp} (preferred: {preferred_name})", (20, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            WHITE = (255, 255, 255)
            bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}"
            cv2.putText(display_img, bot_str, (20, DISPLAY_HEIGHT - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

            det_str = f"Objects Detected: {len(current_dets)}"
            cv2.putText(display_img, det_str, (20, DISPLAY_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("QCar View", display_img)

            k = cv2.waitKeyEx(10)
            handle_keypress(k)

            qcar.write(state.throttle, state.steering)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("--- STARTING QCAR SYSTEM ---")
    pathlib.WindowsPath = pathlib.PosixPath

    print(">> Loading YOLOv5 Model. Please wait...")
    model = load_model()
    if model is None:
        print("[FATAL] Could not load model. Exiting.")
        exit(1)

    print(">> Warming up model...")
    dummy_img = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
    infer_on_frame(model, dummy_img)

    vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
    vision_thread.start()

    control_thread = Thread(target=controlLoop)
    control_thread.start()

    control_thread.join()
    state.kill = True
    print("--- SYSTEM SHUTDOWN COMPLETE ---")


# # integrated_qcar_control_with_detection.py

# import os
# import signal
# import time
# from threading import Thread, Lock

# import cv2
# import numpy as np
# import pathlib

# from pal.products.qcar import QCar, QCarGPS, QCarCameras
# from pal.utilities.math import wrap_to_pi
# from hal.content.qcar_functions import QCarEKF
# from hal.products.mats import SDCSRoadMap

# from qcar_detector import load_model, infer_on_frame


# # ===================== Experiment Configuration =====================
# tf = 6000
# startDelay = 1
# controllerUpdateRate = 100

# # Speed controller
# v_ref = 0.5
# K_p = 0.2
# K_i = 0.25

# # Steering controller
# enableSteeringControl = True
# K_stanley = 0.3
# nodeSequence = [10, 4, 20, 10]

# # GPS calibration
# calibrate = False
# calibrationPose = [0, 2, -np.pi / 2]

# # Camera / display
# CSI_WIDTH, CSI_HEIGHT = 820, 410
# DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410
# PREFERRED_CAM = "LEFT"        # "FRONT" or "LEFT" or "RIGHT"
# FAILOVER_AFTER_SEC = 1.0

# # YOLO worker pacing (keeps CPU reasonable)
# VISION_SLEEP_IDLE = 0.01


# # ===================== Controllers =====================
# class SpeedController:
#     def __init__(self, kp=0.0, ki=0.0):
#         self.maxThrottle = 0.3
#         self.kp = kp
#         self.ki = ki
#         self.ei = 0.0

#     def update(self, v, v_ref, dt):
#         if dt <= 0:
#             return 0.0
#         e = v_ref - v
#         self.ei += dt * e
#         u = self.kp * e + self.ki * self.ei
#         return float(np.clip(u, -self.maxThrottle, self.maxThrottle))


# class SteeringController:
#     def __init__(self, waypoints, k=1.0, cyclic=True):
#         self.maxSteeringAngle = np.pi / 6
#         self.wp = waypoints
#         self.N = len(waypoints[0, :])
#         self.wpi = 0
#         self.k = k
#         self.cyclic = cyclic
#         self.p_ref = (0, 0)
#         self.th_ref = 0

#     def update(self, p, th, speed):
#         # Avoid dividing by ~0 speed in Stanley term
#         speed = max(float(speed), 1e-3)

#         wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
#         wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

#         v = wp_2 - wp_1
#         v_mag = np.linalg.norm(v)
#         if v_mag < 1e-9:
#             return 0.0

#         v_uv = v / v_mag
#         tangent = np.arctan2(v_uv[1], v_uv[0])

#         s = np.dot(p - wp_1, v_uv)
#         if s >= v_mag:
#             if self.cyclic or self.wpi < self.N - 2:
#                 self.wpi += 1

#         ep = wp_1 + v_uv * s
#         ct = ep - p
#         direction = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

#         ect = np.linalg.norm(ct) * np.sign(direction)
#         psi = wrap_to_pi(tangent - th)

#         self.p_ref = ep
#         self.th_ref = tangent

#         delta = wrap_to_pi(psi + np.arctan2(self.k * ect, speed))
#         return float(np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle))


# # ===================== Shared State =====================
# class State:
#     def __init__(self):
#         self.kill = False
#         self.lock = Lock()

#         # Latest camera frame (BGR uint8)
#         self.raw_frame = None
#         self.cam_name = "NONE"

#         # Latest detections list
#         self.latest_detections = []

# state = State()


# def sig_handler(*args):
#     state.kill = True

# signal.signal(signal.SIGINT, sig_handler)


# # ===================== Camera helpers =====================
# def get_stream(cameras, name: str):
#     name = name.upper()
#     if name == "FRONT":
#         return getattr(cameras, "csiFront", None)
#     if name == "LEFT":
#         return getattr(cameras, "csiLeft", None)
#     if name == "RIGHT":
#         return getattr(cameras, "csiRight", None)
#     return None


# # ===================== Vision Thread =====================
# def yolo_worker(yolo_model):
#     print(">> Vision thread started.")
#     while not state.kill:
#         with state.lock:
#             frame = state.raw_frame.copy() if state.raw_frame is not None else None

#         if frame is None:
#             time.sleep(VISION_SLEEP_IDLE)
#             continue

#         # Your infer function should return: (annotated_frame, dets)
#         # We'll use dets and do drawing in the main thread for display stability.
#         _, dets = infer_on_frame(yolo_model, frame)

#         with state.lock:
#             state.latest_detections = dets


# # ===================== Integrated Control + Camera + Display =====================
# def control_and_display_loop(model):
#     # ---- Roadmap / initial pose ----
#     if enableSteeringControl:
#         roadmap = SDCSRoadMap(leftHandTraffic=False)
#         waypointSequence = roadmap.generate_path(nodeSequence)
#         initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
#     else:
#         waypointSequence = None
#         initialPose = [0, 0, 0]

#     speedController = SpeedController(kp=K_p, ki=K_i)

#     if enableSteeringControl:
#         steeringController = SteeringController(
#             waypoints=waypointSequence,
#             k=K_stanley
#         )

#     # ---- QCar I/O ----
#     qcar = QCar(readMode=1, frequency=controllerUpdateRate)

#     # GPS/EKF used for steering control (and also okay if you just want pose logging)
#     ekf = None
#     gps = None
#     if enableSteeringControl or calibrate:
#         ekf = QCarEKF(x_0=initialPose)
#         gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)

#     # Cameras
#     cameras = QCarCameras(
#         frameWidth=CSI_WIDTH,
#         frameHeight=CSI_HEIGHT,
#         frameRate=30,
#         enableFront=True,
#         enableLeft=True,
#         enableRight=True,
#     )

#     # OpenCV window
#     cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)

#     preferred_name = PREFERRED_CAM.upper()
#     active_name = preferred_name
#     last_good_frame_time = time.time()

#     u = 0.0
#     delta = 0.0

#     # ---- Run ----
#     if gps is None:
#         # still allow context manager usage
#         gps_cm = memoryview(b'')
#     else:
#         gps_cm = gps

#     with qcar, cameras, gps_cm:
#         print(">> Control+Display loop running...")
#         t0 = time.time()
#         t = 0.0

#         while (t < tf + startDelay) and (not state.kill):
#             # timing
#             tp = t
#             t = time.time() - t0
#             dt = t - tp

#             # read car sensors
#             qcar.read()
#             v = float(qcar.motorTach)

#             # ---- pose estimate for steering ----
#             if enableSteeringControl:
#                 if gps is not None and gps.readGPS():
#                     y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
#                     ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
#                 else:
#                     ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

#                 x = float(ekf.x_hat[0, 0])
#                 y = float(ekf.x_hat[1, 0])
#                 th = float(ekf.x_hat[2, 0])

#                 # look-ahead point for Stanley
#                 p = (np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2)

#             # ---- camera read (single stream, optional failover) ----
#             img = None
#             active_stream = get_stream(cameras, active_name)
#             if active_stream is not None and active_stream.read():
#                 img = active_stream.imageData
#                 last_good_frame_time = time.time()

#             if img is None and (time.time() - last_good_frame_time) > FAILOVER_AFTER_SEC:
#                 for candidate in ["FRONT", "LEFT", "RIGHT"]:
#                     stream = get_stream(cameras, candidate)
#                     if stream is not None and stream.read():
#                         img = stream.imageData
#                         active_name = candidate
#                         last_good_frame_time = time.time()
#                         break

#             if img is not None:
#                 with state.lock:
#                     state.raw_frame = img.copy()
#                     state.cam_name = active_name

#             # ---- control update ----
#             if t < startDelay or dt <= 0:
#                 u = 0.0
#                 delta = 0.0
#             else:
#                 u = speedController.update(v, v_ref, dt)
#                 if enableSteeringControl:
#                     delta = steeringController.update(p, th, v)
#                 else:
#                     delta = 0.0

#             qcar.write(u, delta)

#             # ---- display ----
#             with state.lock:
#                 display_img = state.raw_frame.copy() if state.raw_frame is not None else None
#                 dets = list(state.latest_detections)
#                 cam_name_disp = state.cam_name

#             if display_img is None:
#                 display_img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

#             # Draw detections
#             for det in dets:
#                 x1, y1, x2, y2, conf, cls_id, label = det
#                 cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 label_text = f"{label} {conf:.2f}"
#                 text_y = y1 - 10 if y1 > 20 else y1 + 20
#                 cv2.putText(display_img, label_text, (x1, text_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#             # UI overlays
#             WHITE = (255, 255, 255)
#             cv2.putText(display_img, f"CAM: {cam_name_disp} (preferred: {preferred_name})", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             cv2.putText(display_img, f"THR: {u:.2f}  STR: {delta:.2f}  v:{v:.2f}", (20, DISPLAY_HEIGHT - 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

#             cv2.putText(display_img, f"Objects Detected: {len(dets)}", (20, DISPLAY_HEIGHT - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#             cv2.imshow("QCar View", display_img)

#             k = cv2.waitKeyEx(1)
#             if k in (ord('q'), ord('Q'), 27):
#                 state.kill = True

#         # stop car
#         qcar.read_write_std(throttle=0, steering=0)

#     cv2.destroyAllWindows()
#     print(">> Control+Display loop exited.")


# # ===================== Main =====================
# if __name__ == "__main__":
#     print("--- STARTING INTEGRATED QCAR CONTROL + DETECTION ---")
#     pathlib.WindowsPath = pathlib.PosixPath

#     print(">> Loading YOLO model...")
#     model = load_model()
#     if model is None:
#         print("[FATAL] Could not load model. Exiting.")
#         raise SystemExit(1)

#     print(">> Warming up model...")
#     dummy = np.zeros((CSI_HEIGHT, CSI_WIDTH, 3), dtype=np.uint8)
#     infer_on_frame(model, dummy)

#     # Start vision thread
#     vision_thread = Thread(target=yolo_worker, args=(model,), daemon=True)
#     vision_thread.start()

#     # Run integrated loop (control + camera + display)
#     try:
#         control_and_display_loop(model)
#     finally:
#         state.kill = True

#     print("--- SYSTEM SHUTDOWN COMPLETE ---")