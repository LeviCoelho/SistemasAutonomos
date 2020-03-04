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
    
    DIR = "/home/levi/Pictures"
    # initialize the list of image URLs to download
    urls = "http://192.168.0.102:8080/shot.jpg"
    # download the image URL and display it
    print("Diretorio em que a imagem sera salva: %s" % DIR)
    print("URL da imagem: %s" % urls)
    default = input("Usar dados acima? [y,n]")
    if default == 'Y' or default == 'y':
        print ('downloading %s' % urls)
        image = url_to_image(urls)
        Save(DIR,image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    if default == 'N' or default == 'n':
        urls = input("Informe a URL da imagem: ")
        DIR = input("Informw o Diretorio em que a imagem sera salva: ")
        print ('downloading %s' % urls)
        image = url_to_image(urls)
        Save(DIR,image)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
    
if __name__ == '__main__':
    main()