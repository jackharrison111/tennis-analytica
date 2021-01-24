

import torchvision
from torchvision import transforms
import cv2
import numpy as np
from SourceCode.VideoReader import VideoReader
import time
import csv



start = time.perf_counter()


filename = "StereoClips/stereoClip1_Kyle.mov"

filename2 = "HomeClips/Dad/behindFence/middleRally2_Dad.mp4"
filename3 = "HomeClips/edited/Dad/rally1-edit_Dad.mp4"

vr1 = VideoReader(filename=filename3)
print(vr1.numFrames)

frameskip = 65
vr2 = VideoReader(filename=filename3)
vr2.setNextFrame(frameskip)

fourcc = cv2.VideoWriter_fourcc(*'mp4v');
testOutputVid = cv2.VideoWriter('UntrackedFiles/RCNNattempt1.mp4',fourcc, 30.0, (1920,1080));



coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# create a retinanet inference model
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model.eval()

prott1 = r'vision_models/MobileNetSSD_deploy.prototxt'
prott2 = r'vision_models/mobilenet_iter_73000.caffemodel'
model = cv2.dnn.readNetFromCaffe(prott1, prott2)

ret, frame = vr1.readFrame()
blob = cv2.dnn.blobFromImage(frame,0.007843, (300, 300), 127.5)

blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),1, (300, 300))

model.setInput(blob)
start = time.perf_counter()
detections = model.forward()
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")


csvData = []
frameNum = 0
while(True):

    vr1.setNextFrame(frameNum)
    ret1, frame1 = vr1.readFrame()

    image = frame1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not (ret1):
        print('Ending after', frameNum - 1, 'frames.')
        break;

    #if frameNum > 400:

    #break

    csvTuple = []
    csvTuple.append(frameNum)
    # predict detections in the input image
    image_as_tensor = transforms.Compose([transforms.ToTensor(), ])(image)
    outputs = model(image_as_tensor.unsqueeze(0))

# post-process the detections ( filter them out by score )
    detection_threshold = 0.5
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels']
#boxes returned in form [x1,y1, x2,y2]
    box_centers = [(int(box[0]+(box[2]- box[0])/2), int(box[1] +(box[3]-box[1])/2)) for box in boxes]

    for i, box in enumerate(box_centers):
        if pred_classes[i] in ['person', 'sports ball', 'frisbee', 'tennis racket']:
            cv2.circle(image, box, 5, COLORS[labels[i]], 3)

        if pred_classes[i] == 'person':
            csvTuple.append(box)

    testOutputVid.write(image);
    print(frameNum)
    csvData.append(list(csvTuple))
    frameNum+=1

testOutputVid.release()
end = time.perf_counter()
print(f"Done {frameNum} frames in {end - start}s.")

with open('UntrackedFiles/playerData2Grey.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerows(csvData)



'''
# draw predictions
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
for i, box in enumerate(boxes):
    color = COLORS[labels[i]]
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                lineType=cv2.LINE_AA)
cv2.imshow('Image', image)
cv2.waitKey(0)
'''