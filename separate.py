import numpy as np
import cv2
import time

cap = cv2.VideoCapture('video file')
start = time.time()
name = 1

while(cap.isOpened()):
	ret, frame = cap.read()

	cv2.imshow('frame',frame)
	elapsed = time.time() - start

	if elapsed >= 0.2:
		
		start = time.time()
		cv2.imwrite('C:/Matei/Python/TENNISBALL/MOREADVANCED/video/' + str(name) + '.png', frame)
		name += 1

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
