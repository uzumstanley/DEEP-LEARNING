import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms

# COCO dataset class names
classes = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
    'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
    'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

# load the pytorch model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# set the model in evaluation mode
model.eval()

# transform the input to tensor
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# random color for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    nn_input = transform(frame)
    output = model([nn_input])
    
    # iterate over the network output for all boxes
    for box, box_class, score in zip(output[0]['boxes'].detach().numpy(),
                                     output[0]['labels'].detach().numpy(),
                                     output[0]['scores'].detach().numpy()):

        # filter the boxes by score
        if score > 0.5:
            # transform bounding box format
            box = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]

            # select class color
            color = colors[box_class]

            # extract class name
            class_name = classes[box_class]

            # draw the bounding box
            cv2.rectangle(img=frame,
                          pt1=box[0],
                          pt2=box[1],
                          color=color,
                          thickness=2)

            # display the box class label
            cv2.putText(img=frame,
                        text=class_name,
                        org=box[0],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=2)
 
    
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()