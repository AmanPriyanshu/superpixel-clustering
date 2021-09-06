import cv2
import numpy as np
import os

path = './images/'
images = np.stack([cv2.imread(path+i) for i in os.listdir(path)])
print(images.shape)
video = cv2.VideoWriter('record.mp4', 0, 40, (images.shape[2], images.shape[1])) 
for image in images:
	video.write(image) 
cv2.destroyAllWindows() 
video.release() 