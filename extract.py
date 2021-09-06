from skimage import io
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

image = img_as_float(io.imread("image.png"))
N = image.shape[0]*image.shape[1]
total_time = 5
fps = 30

p = np.exp(np.log(N)/(fps*total_time))

array = []
while N>0:
	numSegments = N
	N = N//p
	array.append(numSegments)

for idx, numSegments in enumerate(array[10:]):
	img = image.copy()
	segments = slic(image, n_segments = int(numSegments), sigma=5, start_label=1)
	for label in tqdm(np.unique(segments), desc=str({"N": int(numSegments)})):
		indexes = np.argwhere(segments==label)
		arr = []
		for (i, j) in indexes:
			arr.append(image[i, j, :])
		arr = np.stack(arr)
		color = np.mean(arr, 0)
		for (i, j) in indexes:
			img[i, j, :] = color
	img = img*255
	img = img.astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	output_path = "./images/"+"0"*(len(str(len(array)))-len(str(idx)))+str(len(array)-idx)+".png"
	cv2.imwrite(output_path, img)