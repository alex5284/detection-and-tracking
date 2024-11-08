import cv2
import sys

tracker = cv2.TrackerMIL_create()
video = cv2.VideoCapture('1.mp4')

if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()
    

frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
scaled_width = int(width * 0.6)
scaled_height = int(height * 0.6)
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (scaled_width, scaled_height))

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    timer = cv2.getTickCount()
    ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2)
    else :
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    cv2.putText(frame,"Tracker: MIL", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    
    cv2.imshow("Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
