import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov10n.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
tracker = Tracker()
count = 0
cap = cv2.VideoCapture('edited480.mp4')

down = {}
counter_down = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    results = model.predict(frame)

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    print(px)

    list = []

    for index, row in px.iterrows():
        #        print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    # print(bbox_id)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        #cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        #cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        #cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        x = 315
        offset = 20

        ''' condition for red line '''
        if x < (cx + offset) and x > (cx - offset):
            down[id] = cx
            if id in down:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                counter_down.add(id)
    print(down)
    # line
    text_color = (255, 255, 255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)

    # print(down)
    cv2.line(frame, (200, 340), (345, 220), red_color, 3)  # starting cordinates and end of line cordinates
    #cv2.putText(frame, ('red line'), (280, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    downwards = (len(counter_down))
    cv2.putText(frame, ('going down - ' + str(len(counter_down))), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1,
                cv2.LINE_AA)

    # cv2.line(frame,(282,308),(1004,308),red_color,3)  #  starting cordinates and end of line cordinates
    # cv2.putText(frame,('red line'),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()