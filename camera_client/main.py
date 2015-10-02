"""
        Standard packadges
"""
import subprocess
import time
import cv2
import numpy as np
import requests
import json

"""
        Own Modules
"""
import camera
UNPROCESSED_FOLDER = 'images_unprocessed'
SUCCESS_FOLDER = 'image_success'
CAMERALOCATION = 'PYCONZA2015'
SERVER_LOC = 'http://127.0.0.1:8080'
cam = camera.ipCamera('http://192.168.1.104/video4.mjpg',None,None)

i=0
while True:
    #print("Save file",i)
    frame = cam.get_frame()
#     cv2.imshow('Test Image',frame)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()
    t0 = time.time()
    imagename = str(i)+'.jpeg'
    imagelocation = 'images_unprocessed/'+imagename
    cv2.imwrite(imagelocation, frame)

        #p = subprocess.Popen(["python", "adaptive_thresholding.py","-i",imagelocation],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        #output = p.communicate()[0]
        #print (output)


    print("poes pas")
    
    files = {'myFile': ( imagename, open(imagelocation, 'rb') )}

    
    r = requests.post(SERVER_LOC +'/pushImage',data=files)
    print(r.status_code, r.reason)
    print("Time to process: ",time.time()-t0)
    i+=1
#python adaptive_thresholding.py -i test_data/2382709.jpg 

