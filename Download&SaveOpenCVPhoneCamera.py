# import the necessary packages
import numpy as np
import urllib.request
import os 
import time
import cv2

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def Save(directory,image):
	timestamp = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
	filename = "%s/%s.jpg" % (directory, timestamp)
	# salvando a imagem
	os.chdir(directory)
	cv2.imwrite(filename, image) 
	print ("Salvo")


def main():
	Q = 0
	DIR = "/home/levi/Pictures/Teste"    
	#IP = input('Informe o IP do dispositivo: ')
	#urls = 'http://' + IP + ':8080/video'
	urls = 'http://10.50.63.247:8080/shot.jpg'	
    # download the image URL and display it
	fim = True	
	while fim:
		fim = True
		#time.sleep(5)# measure process time
		print ('downloading %s' % urls)
		image = url_to_image(urls)
		Save(DIR,image)
		Q = Q + 1
		print(Q)
		if Q == 100:
			fim = True
        
    
if __name__ == '__main__':
	main()
