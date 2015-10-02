import base64
import time
import urllib2
import cv2
import numpy as np
import os

UNPROCESSED_FOLDER = 'images_unprocessed'
SUCCESS_FOLDER = 'image_success'

class ipCamera(object):

    def __init__(self, url, user=None, password=None):
        print("INIT IP CAMERA")
        self.prevframe = None
        self.currframe = None
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]
        self.req = urllib2.Request(self.url)
        if user != None and password != None:
                self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        bytes = ''
        response = urllib2.urlopen(self.req)
        done = False
        while not done:
            bytes+=response.read(1024)
            a = bytes.find('\xff\xd8')
            b = bytes.find('\xff\xd9')
            if a!=-1 and b!=-1:
                jpg = bytes[a:b+2]
                bytes= bytes[b+2:]
                frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
                #cv2.imshow('Test Image',frame)
                done = True
        self._set_prevframe(self.currframe)
        self.currframe = frame
        return frame

    def _set_prevframe(self,frame):
        self.prevframe = frame
