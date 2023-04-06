import cv2
from cv2 import resize
import numpy as np
from PIL import Image
import io
from matplotlib import pyplot as plt
# from traceback2 import print_tb
import dlib

from id_card_recognition.utils import rotate_bbox, rotate_bound, warpImg, findFaces, is_two_image_same
from id_card_recognition.utils import resizeImage, get_warpPerspective, get_angle_and_box_coord


def siftMatching(img1, img2):
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print("Total good matches:", len(good))
    good = good[:20]
    return kp1, kp2, good


def sift(img):
    template = img
    # sample = cv2.imread("id_card_recognition/train/test.png")
    # MIN_MATCH_COUNT = 20

    img1 = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    img1 = resizeImage(img1)
    # kp1, kp2, good = siftMatching(img1, img2)
    face_crop_img_query = findFaces(img1)
    return face_crop_img_query
    # if len(good) >= MIN_MATCH_COUNT:

    #     face_crop_img_query = findFaces(img1)
    #     return 'ID_Card', face_crop_img_query

    # else:
    #     return 'Not_ID_Card', findFaces(img1)

def match(img1, img2):
    return is_two_image_same(img1, img2, 15)