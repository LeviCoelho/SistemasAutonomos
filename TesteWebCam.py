#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 07:52:54 2020

@author: levi
"""

#! /usr/bin/python
 
import pygame, sys
import pygame.camera
import time

WEBCAM_DIR = "/home/levi/Pictures"
 
pygame.init()
pygame.camera.init()
cam = pygame.camera.Camera("/dev/video0", (640,480))
# pegando foto da webcam
cam.start()
image = cam.get_image()
cam.stop
 
timestamp = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
filename = "%s/%s.jpg" % (WEBCAM_DIR, timestamp)
 
# salvando a imagem
pygame.image.save(image, filename)
 
print ("Salvo")