#/usr/bin/python

import argparse
import glob
import math
import cv2
import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage.filter import threshold_adaptive
from pyfann import libfann

import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

#NOTE: Stolen from pyimagesearch, check out Adrian's tutorials and examples

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

 
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped


#NOTE: My own quick and dirty attempt at letter classification

img_set = {}
for fn in glob.glob("Averages/*.png"):
    img_set[fn.split(".")[0][-1]] = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)

def match_against_average_chars(img):
    resultset = {}
    # I am sure there is a better way, but this number should be bigger than any MSE
    minval = 9999999999999
    minchar = '-'
    for char in img_set:
        resultset[char] = mse(img, img_set[char])
        if resultset[char]<minval:
            minval = resultset[char]
            minchar = char
    return minchar, resultset

ann = libfann.neural_net()
ann.create_from_file("n251.net")
def match_with_neural_net(img):
    #print "a"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    a=[]
    [a.extend(x) for x in img]
    #print a
    result = ann.run([float(x) for x in a])
    
    #print letters[result.index(max(result))], max(result)
    #cv2.imshow("CHAR", img)
    #cv2.waitKey(0)

    return letters[result.index(max(result))], {}

class Detector():
    def __init__(self, config=None, image=None, debug=False):
        if not config:
            # Default na iets wat wat min of meer sal werk
            config = {"y_offset": 20, # maximum y offset between chars
                    "x_offset":  55, # maximum x gap between chars
                    "thesh_offset":  0, # this determines the cutoff point on the adaptive threshold.
                    "thesh_window": 25, # window of adaptive theshold area
                    # max min char width, height and ratio
                    "w_min":  6, # char pixel width min
                    "w_max":  30, # char pixel width max
                    "h_min":  12, # char pixel height min
                    "h_max":  40, # char pixel height max
                    "hw_min":  1.5, # height to width ration min
                    "hw_max":  3.5, # height to width ration max
                    "h_ave_diff":  1.09,  # acceptable limit for variation between characters
                }
        self.setConfig(config)
        self.plates = []
        self.image = image
        self.thesh = None
        self.debug = debug
    
    def setConfig(self, config):
        self.config = config
    
    def detect_plates(self, image=None, level=1):
        if image is not None:
            self.image = image
        # First convert to black ad white
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # many magic values in here, the thresh offset is around 0, as thresholding is done for a sliding window of 25x25 pixels
        self.thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.config["thesh_window"], self.config["thesh_offset"])
        if self.debug:
            self.allblobs = self.image.copy()
            self.reducedblobs = self.image.copy()
            self.roiblobs = self.image.copy()
        self.image = image
        #self.thesh = thresh
        # now we have a blck and white image and need to find all blobs that match our size and aspect requirements
        # find contours (This acts like a CCA) scikit seems to have a CCA as well, I just happended to find this first.
        # merge requests are welcome... with proof of higher accurace or quicker execution
        (cnts, _) = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours and discard all odd shapes and sizes
        correctly_sized_list = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if self.debug:
                cv2.rectangle(self.allblobs, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            # Filter on width and height
            if self.config["w_min"] < w < self.config["w_max"] and self.config["h_min"] < h < self.config["h_max"] and self.config["hw_min"] < 1.0*h/w < self.config["hw_max"]:
                correctly_sized_list.append((x, y, w, h))
                if self.debug:
                    cv2.rectangle(self.reducedblobs, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        
        # now we try to filter based on character proximity and the fact that they would be in a row
        self.possible_plate_regions = []
        # sort by x position
        self.sort_list = sorted(correctly_sized_list, key=lambda x: x[0])

        # Try to group blobs into platelike groups
        for char in self.sort_list:
            placed_char = False
            # Check if this blob has same y and x within offset values off current are (this is why we sorted by x value).
            for region in self.possible_plate_regions:
                if region[-1][1] - self.config["y_offset"] < char[1] < region[-1][1] + self.config["y_offset"] and region[-1][0] + self.config["x_offset"] > char[0]:
                    region.append(char)
                    placed_char = True
                    break
            # if char was not placed in a group, it becomes the first of a new group
            if placed_char is False:
                self.possible_plate_regions.append([char])

        # Now remove chars from regions if heights differ significantly, as numberplate chars are evenly sized. This could possibly be done in above filter, but this seemed better
        self.possible_plate_regions_ave_filtered = []

        for region in self.possible_plate_regions:
            if len(region)>2:
                self.possible_plate_regions_ave_filtered.append([])
                ave = sum([char[3] for char in region])/len(region)
                for char in region:
                    if ave/self.config["h_ave_diff"] < char[3] < ave*self.config["h_ave_diff"]:
                        self.possible_plate_regions_ave_filtered[-1].append(char)
        
        # Now filter char regions on count
        self.possible_plate_regions_ave_filtered = [x for x in self.possible_plate_regions_ave_filtered if len(x)>2]

        possible_plate_regions_plate_details = []
        
        for region in self.possible_plate_regions_ave_filtered:
            # Find the min and max values of the plate region
            xmin = min([x[0] for x in region])
            ymin = min([x[1] for x in region])
            xmax = max([x[0]+x[2] for x in region])
            ymax = max([x[1]+x[3] for x in region])
            topleft = sorted(region, key=lambda x: x[0]+x[1])[0]
            topright = sorted(region, key=lambda x: -(x[0]+x[2])+x[1])[0]
            botleft = sorted(region, key=lambda x: x[0]-(x[1]+x[3]))[0]
            botright = sorted(region, key=lambda x: -(x[0]+x[2])-(x[1]+x[3]))[0]
            
            #print (topleft, topright, botleft, botright)
            
            mtop = 1.0*(topleft[1]-topright[1])/(topleft[0]-(topright[0]+topright[2]))
            mbot = 1.0*(botleft[1]+botleft[3]-(botright[1]+botright[3]))/(botleft[0]-(botright[0]+botright[2]))
            #print mtop, mbot
            if self.debug:
                for char in region:
                    (x, y, w, h) = char
                    cv2.rectangle(self.roiblobs, (x, y), (x + w, y + h), (0, 0, 255), 1) 
            
            possible_plate_regions_plate_details.append({"size": (xmin, ymin, xmax, ymax), 
                                                         "roi": (xmin - 2*self.config["w_max"], ymin - self.config["h_max"], xmax + 2*self.config["w_max"], ymax + self.config["h_max"]),
                                                         "average_angle": (mtop + mbot)/2.0})
            # Get area plus 2 x max char width to the sides and max char height above and below
            try:
                self.skew_correct(possible_plate_regions_plate_details[-1])
                
                # use thresholded roi to find chars again
                if "warped2" in possible_plate_regions_plate_details[-1] and possible_plate_regions_plate_details[-1]["warped2"] is not None:
                    self.detect_chars(possible_plate_regions_plate_details[-1])
                    if len(possible_plate_regions_plate_details[-1]["plate"])>3:
                        possible_plate_regions_plate_details[-1]["somechars"] = True
            except Exception as ex:
                print ex
            
        self.plates = possible_plate_regions_plate_details
        return self.plates
            
    def detect_chars(self, plate_detail):
        (cnts, _) = cv2.findContours(plate_detail["warped2"].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        plate_detail["plate"] = ""
        plate_detail["chars"] = []
        #plate_detail["char_image"] = []
        #plate_detail["char_image_scaled"] = []
        #plate_detail["char_image_scaled"] = []
        sorted_char_list = sorted([cv2.boundingRect(c) for c in cnts], key=lambda x: x[0])
        short_sorted_char_list = []
        for (x, y, w, h) in sorted_char_list:
            if self.config["w_min"] < w < self.config["w_max"] and self.config["h_min"] < h < self.config["h_max"] and self.config["hw_min"] < 1.0*h/w < self.config["hw_max"]:
                short_sorted_char_list.append((x, y, w, h))
        #ave = sum([char[3] for char in short_sorted_char_list])/len(short_sorted_char_list)
        #mseval = [abs(sum([(char[3]-outer[3]) for char in short_sorted_char_list])) for outer in short_sorted_char_list]
        ave = most_common([bla[3] for bla in short_sorted_char_list]) #[mseval.index(min(mseval))][3]
        #print [bla[3] for bla in short_sorted_char_list]
        #print ave
        #miny = min([char[1] for char in short_sorted_char_list])/len(short_sorted_char_list)
        #maxh = max([char[3] for char in short_sorted_char_list])/len(short_sorted_char_list)
        for (x, y, w, h) in short_sorted_char_list:
            if  ave/self.config["h_ave_diff"] < h < ave*self.config["h_ave_diff"]:
                character = plate_detail["warped"][y-1:y + h+1, x:x + w].copy()
                chardict = {} 
                chardict["char_image"] = character
                resized = cv2.resize(character, (30, 30), interpolation=cv2.INTER_CUBIC)#cv2.equalizeHist(cv2.resize(character, (30, 30), interpolation=cv2.INTER_CUBIC))
                chardict["char_image_scaled"] = resized
                
                chardict["text"], chardict["match_result_dict"] = match_with_neural_net(resized)
                if chardict["text"] == 'Q':
                    chardict["ave_text"], chardict["ave_match_result_dict"] = match_against_average_chars(resized)
                    plate_detail["plate"] += chardict["ave_text"]
                    #print "AVE GEBRUIK", plate_detail["plate"]
                else:
                    #print chardict["text"]
                    plate_detail["plate"] += chardict["text"]
                plate_detail["chars"].append(chardict)
            #else:
                #print "discard op ave", (x, y, w, h)
    
    def skew_correct(self, plate_detail, chars=None):
        (xmin, ymin, xmax, ymax) = plate_detail["size"]
        if True:
            # rotate our roiblobs
            plateregion = self.image[ymin - self.config["h_max"]:ymax + self.config["h_max"], xmin - 2*self.config["w_max"]:xmax + 2*self.config["w_max"]].copy()
            (h, w) = plateregion.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            degrees = math.atan(plate_detail["average_angle"]) * 180 / math.pi
            #print degrees
            #rotate_deg = - atan(((*glob_charlys_lys_it).m1+(*glob_charlys_lys_it).m2)/2.0) * 180.0 / 3.14159265;
            #degrees = 0 # som om die gemiddelde M te kry en dan skakel ons dit om na degree
            M = cv2.getRotationMatrix2D((cX, cY), degrees, 1.0)
            rotated = cv2.warpAffine(plateregion, M, (w, h))
            #cv2.imshow("Rotated by xx Degrees", rotated)
            #cv2.waitKey(0)
            plate_detail["warped"] = rotated
            warped2 = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.config["thesh_window"], self.config["thesh_offset"])
            plate_detail["warped2"] = warped2
        #TODO: Rotate eerder as correct
        else:
            plate_detail["plate"] = ""
            plateregion = self.image[ymin - self.config["h_max"]:ymax + self.config["h_max"], xmin - 2*self.config["w_max"]:xmax + 2*self.config["w_max"]].copy()
            if self.debug:
                cv2.rectangle(self.roiblobs, (xmin - 2*self.config["w_max"], ymin - self.config["h_max"]), (xmax + 2*self.config["w_max"], ymax + self.config["h_max"]), (255, 0, 0), 1)
            #gray = plateregion#cv2.cvtColor(plateregion, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(plateregion, (5, 5), 0)
            edged = cv2.Canny(gray, 25, 250)
            #edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, thesh_offset)
            (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
            screenCnt = None
            plate_detail["edged"] = edged
            # find contours, stolen from pyimagesearch
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                cv2.drawContours(plateregion, [approx], -1, (0, 0, 0), 1)
                if len(approx) == 4:
                    screenCnt = approx
                    break
            plate_detail["screenCnt"] = screenCnt
            if plate_detail["screenCnt"] is not None:
                warped = four_point_transform(plateregion, screenCnt.reshape(4, 2) * 1)
                #cv2.imshow("warped2", warped)
                #cv2.waitKey(0)
                warped2 = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.config["thesh_window"], self.config["thesh_offset"])
                plate_detail["warped"] = warped
                plate_detail["warped2"] = warped2
            return plate_detail
    

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Path to the image")
    ap.add_argument("-c", "--csv", required=False, help="File with CSV test data")
    ap.add_argument("-d", "--debug", required=False, help="Show images set to True")
    args = vars(ap.parse_args())
    if args["image"]:
        image = cv2.imread(args["image"])
        ob = Detector(debug=True if args["debug"] == "True" else False)
        
        retval = ob.detect_plates(image=image)
        print " ".join([x["plate"] if "plate" in x else "" for x in retval])
        if args["debug"] == "True":
            cv2.imshow("Blobs ALL", ob.allblobs)
            cv2.imshow("Blobs size filter", ob.reducedblobs)
            cv2.imshow("Blobs group filtered", ob.roiblobs)
            cv2.waitKey(0)
            for detail in retval:
                print "PLATE:", detail["plate"] if "plate" in detail else ""
                if "edge" in detail:
                    cv2.imshow("edged"+detail["plate"], detail["edged"])
                if "warped" in detail:
                    cv2.imshow("warped"+detail["plate"], detail["warped"])
                if "warped2" in detail:
                    cv2.imshow("warped2"+detail["plate"], detail["warped2"])
                cv2.waitKey(0)
    else:
        # testdata with images
        if "csv" in args and args["csv"]:
            csv = [x.split(",") for x in open(args["csv"]).readlines()]
            ob = Detector(debug=True)
            hit = 0
            cnt = 0
            for line in csv:
                image = cv2.imread("test_data/%s.jpg"%(line[0]))
                #ob.detect_plates(image=image)
                try:
                    retval = ob.detect_plates(image=image)
                    cnt += 1
                    if line[2] in [x["plate"] for x in retval]:
                        hit += 1
                        print "hit:", line[0], line[2], " ".join([x["plate"] for x in retval])
                        print "CNT", cnt, hit
                        cv2.imwrite(line[0] + line[2] + ".jpg",image )
                        cv2.imwrite(line[0] + line[2] + "_all_blob.jpg", ob.allblobs)
                        cv2.imwrite(line[0] + line[2] + "_reduced_blob.jpg", ob.reducedblobs)
                        cv2.imwrite(line[0] + line[2] + "_roi_blob.jpg", ob.roiblobs)
                    #else:
                        #print "mis:", line[0], line[2], " ".join([x["plate"] for x in retval])
                except:
                    print "misex:", line[0]
                #cv2.imshow("Blobs ALL", ob.allblobs)
                #cv2.imshow("Blobs size filter", ob.reducedblobs)
                #cv2.imshow("Blobs group filtered", ob.roiblobs)
                #cv2.waitKey(0)
                if args["debug"] == "True":
                    for detail in retval:
                        print detail["plate"] if "plate" in detail else ""
                        #if "edge" in detail:
                            #cv2.imshow("edged"+detail["plate"], detail["edged"])
                        if "warped" in detail:
                            cv2.imshow("warped"+detail["plate"], detail["warped"])
                        #if "warped2" in detail:
                            #cv2.imshow("warped2"+detail["plate"], detail["warped2"])
                    cv2.imshow("Blobs group filtered", ob.roiblobs)
                    cv2.waitKey(0)
            print "total:", 100.0*hit/len(csv)






