import cv2
import sys
import numpy as np
from util import *
from cv_operations import *

def find_contours(image_file):
    image = cv2.imread( image_file ) # BGR
    imshow_opencv( image ) # show the original image first
    # canvas = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    imgray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    ret, thresh = cv2.threshold( imgray, 127, 255, 0 )
    contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

    # contours, hierarchy = cv2.findContours( canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    # print( "Contours: %d" % len(contours) )
    # print( "Hierarchy: ", len(hierarchy) )

    # cnt = contours[709]
    # print contours[709]
    cv2.drawContours( image, contours, -1, MINT, 2 ) # draw all found contours
    imshow_opencv( image )

def get_img( image_file ):
    image = cv2.imread( image_file ) # BGR
    #imshow_matplot( image ) # show the original image first
    return image
    # imshow_opencv( image )
    #save_img( 'macaron_bgr.jpg', image )

'''
@Params:
* img: Any (Gray-scale or coloured) image.
* top_left: Top left point tuple
* bottom_right: Bottom right point tuple

@Returns:
* corners: Detected corners as features.
'''
def finding_the_cut( img, top_left, bottom_right ):
    # height and width of the org. image
    height, width = img.shape[0], img.shape[1]
    # ROI coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2-x1, y2-y1

    # remaining_rect = 0
    #
    # # This means the chosen sub-part is a corner piece.
    # if x1 == 0 or x2 == (width-1) /
    #     or y1 == 0 or y2 == (height-1):
    #     remaining_rect = 2
    # else:
    #     remaining_rect = 4

    # alpha_ch_img = combine_images( \
    #                     draw_rectangle( img, h=height//2 ),
    #                     img )
    # imshow_opencv( alpha_ch_img )

    ROI = img[y1:y1+h, x1:x1+w]

    curtain = combine_images( \
                        draw_rectangle( img, w=width, h=height ),
                        img )
    curtain[y1:y1+h, x1:x1+w] = ROI
    imshow_opencv( curtain )

    whole_img = cv2.GaussianBlur( img, (51,51), 0 )

    # Insert ROI back into image
    whole_img[y1:y1+h, x1:x1+w] = ROI

    imshow_opencv( whole_img )

#get_img( sys.argv[1] )
finding_the_cut( get_img( sys.argv[1] ), (202, 440), (350, 576) )
