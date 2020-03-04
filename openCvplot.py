import urllib.request
import cv2
import numpy as np

url = "https://pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png"
url_response = urllib.request.urlopen(url)
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, -1)
cv2.imshow('URL Image', img)
cv2.waitKey()
