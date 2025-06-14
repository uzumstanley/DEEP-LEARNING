import os.path

import cv2  # opencv import
import numpy as np
import requests

print("YOLO-V3 object detection")

# Download YOLO net config file
# We'll it from the YOLO author's github repo
yolo_config = 'yolov3.cfg'
if not os.path.isfile(yolo_config):
    url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
    r = requests.get(url)
    with open(yolo_config, 'wb') as f:
        f.write(r.content)

# Download YOLO net weights
# We'll it from the YOLO author's website
yolo_weights = 'yolov3.weights'
if not os.path.isfile(yolo_weights):
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    r = requests.get(url)
    with open(yolo_weights, 'wb') as f:
        f.write(r.content)

# load the network
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Download class names file
# Contains the names of the classes the network can detect
classes_file = 'coco.names'
if not os.path.isfile(classes_file):
    url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    r = requests.get(url)
    with open(classes_file, 'wb') as f:
        f.write(r.content)

# load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Download object detection image
#image_file = 'source_1.png'
# Load webcam
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  # read and normalize image
   
  blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
  
  # set as input to the net
  net.setInput(blob)
  
  # get network output layers
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  
  # inference
  # the network outputs multiple lists of anchor boxes,
  # one for each detected class
  outs = net.forward(output_layers)
  
  # extract bounding boxes
  class_ids = list()
  confidences = list()
  boxes = list()
  
  # iterate over all classes
  for out in outs:
      # iterate over the anchor boxes for each class
      for detection in out:
          # bounding box
          center_x = int(detection[0] * frame.shape[1])
          center_y = int(detection[1] * frame.shape[0])
          w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
          x, y = center_x - w // 2, center_y - h // 2
          boxes.append([x, y, w, h])
  
          # confidence
          confidences.append(float(detection[4]))
  
          # class
          class_ids.append(np.argmax(detection[5:]))
  
  # non-max suppression
  ids = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.75, nms_threshold=0.5)
  
  # draw the bounding boxes on the image
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  
  # iterate over all boxes
  for i in ids:
       
      x, y, w, h = boxes[i]
      class_id = class_ids[i]
  
      color = colors[class_id]
  
      cv2.rectangle(img=frame,
                    pt1=(round(x), round(y)),
                    pt2=(round(x + w), round(y + h)),
                    color=color,
                    thickness=3)
  
      cv2.putText(img=frame,
                  text=f"{classes[class_id]}: {confidences[i]:.2f}",
                  org=(x - 10, y - 10),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=0.8,
                  color=color,
                  thickness=2)
  
  cv2.imshow("Object detection", frame)
 
    # Press Q on keyboard to  exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()