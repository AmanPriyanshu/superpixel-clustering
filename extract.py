from skimage import io
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2
import numpy as np

image = img_as_float(io.imread("image.png"))
for numSegments in [1000]:
	img = image.copy()
	segments = slic(image, n_segments = numSegments, sigma=5, start_label=1)
	for label in np.unique(segments):
		indexes = np.argwhere(segments==label)
		arr = []
		for (i, j) in indexes:
			arr.append(image[i, j, :])
		arr = np.stack(arr)
		color = np.mean(arr, 0)
		for (i, j) in indexes:
			img[i, j, :] = color
	cv2.imwrite("t.png", img*255)