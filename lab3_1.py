import cv2
import torch
import time
import sys

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video = cv2.VideoCapture("1.mp4")

if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()

fps_start_time = time.time()
fps_frame_count = 0

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(video.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output1.avi', cv2.VideoWriter_fourcc(*'FMP4'), 15, (width, height))

while True:
    ok, frame = video.read()
    if not ok:
        break
    input_tensor = frame

    with torch.no_grad():
        predictions = model([input_tensor])

    boxes = predictions.pandas().xyxy[0]

    for ind, box in boxes.iterrows():
        if box["name"] == "person":
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

    fps_frame_count += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_text = f"FPS: {round(fps, 2)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_frame_count = 0
        fps_start_time = time.time()
    
    out.write(frame)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()