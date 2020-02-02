import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 200
data_dir = '/home/sonu/Desktop/data/train/left'
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    cv2.resize(frame, (256, 256))

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(os.path.join(data_dir, img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1
    print('loop')
cam.release()

cv2.destroyAllWindows()