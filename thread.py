import torch
from threading import Thread
from threading import Lock
import numpy as np
import time
import cv2


def readwebcam():
    cap = cv2.VideoCapture(0)
    global frames

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(frame.shape)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        frame = np.transpose(frame, (2, 0, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        lock.acquire()
        frames = np.delete(frames, (0), axis=0)
        frames = np.append(frames, frame.reshape(1, 3, 480, -1), axis=0)
        lock.release()
        print(torch.unsqueeze(torch.from_numpy(frames), 0).permute(0, 2, 1, 3, 4).size())
        #time.sleep(0.1)
    # When everything done, release the capture
    cap.release()


lock = Lock()
t = Thread(target=readwebcam)
t.start()
frames = np.random.rand(16, 3, 480, 640)
i = 1
while (True):
    lock.acquire()
    print(frames[1])
    lock.release()
    print(i)
    i = i + 1
    time.sleep(1)

t.join()

#    print(num)