"""
Requirements 
- pip install OpenCV-Python MediaPipe imutils
- pip install https://cdn.evilmadscientist.com/dl/ad/public/AxiDraw_API.zip
Resources
- Hand Tracking - https://www.makeuseof.com/python-hand-tracking-opencv/?newsletter_popup=1
- AxiDraw - https://axidraw.com/doc/py_api/#quick-start-interactive-xy 
"""
import cv2
from mediapipe.tasks import python
import mediapipe as mp
from mediapipe.tasks.python import vision
import imutils
import numpy as np
from pyaxidraw import axidraw

print("Starting Hand Tracking...")

BaseOptions = python.BaseOptions
HandLandmarker = vision.FaceLandmarker
HandLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,  # simpler than LIVE_STREAM
    num_faces=1
)

landmarker = HandLandmarker.create_from_options(options)

def process_img(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return result

# Drawing landmark connections
def draw_hand_connections(img, result):
    cx, cy = (-1, -1)

    if result.face_landmarks:
        face = result.face_landmarks[0]
        lm = face[4]  # index fingertip

        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)

        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    return (cx, cy), img
  
# define a video capture object
vid = cv2.VideoCapture(0)

# borders = [(20,12), (480,12), (20, 250), (480, 250)] TL TR BL BR
  
prev_coords = (-1, -1)

ad = axidraw.AxiDraw()
ad.interactive()
if not ad.connect():
    print("Failed to connect to AxiDraw.")
    # quit()

canvas = np.zeros((500, 500, 3), dtype=np.uint8)  # black drawing board
draw_prev = None

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width = 500, height = 500)
    results = process_img(frame)
    coords, annotated_frame = draw_hand_connections(frame, results)
    
    if coords[0] > 0 and coords[1] > 0:
        cx, cy = coords

        # map coordinates
        ix = (cx - 20) * (17/460)
        iy = (cy - 12) * (11/238)

        # DRAW ON CANVAS
        if draw_prev is not None:
            cv2.line(canvas, draw_prev, (cx, cy), (255, 255, 255), 2)

        draw_prev = (cx, cy)

        # AXIDRAW
        print(f"Moving to: ({ix}, {iy})")
        # ad.lineto(ix, iy)
    else:
        draw_prev = None


    prev_coords = coords
  
    # Display the resulting frame
    cv2.imshow('frame', annotated_frame)
    cv2.imshow('Drawing', canvas)
      
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        ad.disconnect()
        break

    if key == 32:  # SPACE
        canvas = np.zeros((500, 500, 3), dtype=np.uint8)
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()