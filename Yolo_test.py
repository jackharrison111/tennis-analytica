
'''
taken from: https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python
'''

import cv2

import numpy as np
from SourceCode.VideoReader import VideoReader
import time
import sys
import os
from sklearn.cluster import KMeans

filename3 = "HomeClips/edited/Dad/rally1-edit_Dad.mp4"
filename = "StereoClips/stereoClip1_Kyle.mov"
vr1 = VideoReader(filename=filename)
ret, frame = vr1.readFrame()


fourcc = cv2.VideoWriter_fourcc(*'mp4v');
testOutputVid = cv2.VideoWriter('UntrackedFiles/Yolo-tiny_kyleclip1.mp4',fourcc, 30.0, (1920,1080));


CONFIDENCE = 0.25
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 1

COLORS = np.random.uniform(0, 255, size=(85, 3))
go = time.perf_counter()
# the neural network configuration
config_path = "object-detection/cfg/yolov3-tiny.cfg"
# the YOLO net weights file
#weights_path = "object-detection/weights/yolov3.weights"
weights_path = "object-detection/weights/yolov3-tiny.weights"

# loading all the class labels (objects)
labels = open("object-detection/data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")


# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)


path_name = "object-detection/images/dog2.jpg"
filename = "UntrackedFiles/Yolo-tiny_test.jpg"
ext= "jpg"
image = cv2.imread(path_name)


frameNum = 0
km = KMeans(n_clusters=2)

while(True):

    vr1.setNextFrame(frameNum)
    ret1, frame1 = vr1.readFrame()

    image = frame1

    if not (ret1):
        print('Ending after', frameNum - 1, 'frames.')
        break;



    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)





    # sets the blob as the input of the network
    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    #print(f"Time took: {time_took:.2f}s")

    font_scale = 1
    thickness = 1
    centers, boxes, confidences, class_ids = [], [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability

            if confidence > CONFIDENCE and class_id == 0:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height

                box = detection[:4] * np.array([w, h, w, h])

                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                centers.append((centerX, centerY))
                confidences.append(float(confidence))
                class_ids.append(class_id)



    if len(centers) > 1:
        km.fit(centers)
        cluster_centers = km.cluster_centers_
    else:
        cluster_centers = centers

    for i, center in enumerate(centers):
        cv2.circle(image, (int(center[0]), int(center[1])), 5, color=list(COLORS[class_ids[i]]), thickness=3)


    testOutputVid.write(image);
    print(frameNum)
    frameNum +=1
    #cv2.imwrite(filename, image)

testOutputVid.release()
end = time.perf_counter()
print(f"Done {frameNum} frames in {end - go}s.")

'''
# loop over the indexes we are keeping
for i in range(len(boxes)):
    # extract the bounding box coordinates
    x, y = boxes[i][0], boxes[i][1]
    w, h = boxes[i][2], boxes[i][3]
    # draw a bounding box rectangle and label on the image
    color = [int(c) for c in colors[class_ids[i]]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
    text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
    # calculate text width & height to draw the transparent boxes as background of the text
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
    text_offset_x = x
    text_offset_y = y - 5
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
    overlay = image.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
    # add opacity (transparency to the box)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    # now put the text (label: confidence %)
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)



# perform the non maximum suppression given the scores defined before
idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

cv2.imwrite(filename + "_yolo3_2." + ext, image)

'''
