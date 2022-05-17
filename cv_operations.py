import cv2
import sys
import numpy as np
from util import *

# BGR Colours
DARK = (0, 0, 0)
LIGHT = (255, 255, 255)
LIGHT_GREY = (224, 224, 224)
MINT = (212, 255, 127)
TURQ = (208, 224, 64)
YELLOW = (0, 255, 255)


'''@Params:
    * img: An image.

    @Returns:
    * cropped_img: Cropped image.
'''
def crop_img( img, x1=0, y1=0, x2=None, y2=None ):
    ''' For x1 and y1,
    if they weren't assigned, use the min. length for them.
    ..::..
    For x2 and y2,
    if they weren't assigned, use the max. length for them.
    '''
    x2 = x2 if x2 is not None else img.shape[1] # width  <-> columns
    y2 = y2 if y2 is not None else img.shape[0] # height <-> rows
    cropped_img = img[ y1:y2, x1:x2 ]
    # imshow_opencv( cropped_img )
    return cropped_img


def add_border( img, top=10, bottom=10, left=10, right=10 ):
    return cv2.copyMakeBorder( img,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=LIGHT )


def insert_img( img1, img2, x1=0, y1=0, x2=None, y2=None ):
    overlay = img1.copy()
    x2 = x2 if x2 is not None else img1.shape[1] # width  <-> columns
    y2 = y2 if y2 is not None else img1.shape[0] # height <-> rows

    overlay[ y1:y2, x1:x2 ] = img2
    return overlay


def combine_images( img1, img2, factor_img1=0.5 ):
    return cv2.addWeighted( img1, factor_img1, img2, 1-factor_img1, 0 )


def draw_rectangle( img, x=0, y=0, w=None, h=None ):
    overlay = img.copy()
    w = w if w is not None else img.shape[1] # width  <-> columns
    h = h if h is not None else img.shape[0] # height <-> rows

    cv2.rectangle( overlay, (x, y), (x+w, y+h), LIGHT, cv2.FILLED )
    return overlay


''' Finds trackable features in an image based on Shi-Tomasi algorithm.
@Params:
* gray_img: Gray-scale image.

@Returns:
* corners: Detected corners as features.
'''
def detect_features( gray_img, max_feat=0 ):
    ''' @Params - goodFeaturesToTrack:
        Mandatory:
            input_image
            max (strongest) corners that will be returned; if<=0 -> return all
            quality_level
            min Euclidean distance between the found corners
        Optional:
            useHarrisDetector = false
    For more info:
        https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html
        https://docs.opencv.org/4.2.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    '''
    # TODO: Try with ORB or FAST to test speed: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
    corners = cv2.goodFeaturesToTrack( gray_img, max_feat, 0.01, 10 )
    corners = np.int0( corners )    # Turns all the elements into type int64.
    return corners


def orb( img ):
    ''' Initiate ORB detector
    @Params:
        scoreType - options:
            cv2.ORB_FAST_SCORE -> 1; this one finds more features, yet some of them are not accurate.
            cv2.ORB_HARRIS_SCORE -> 0
    '''
    orb = cv2.ORB_create( scoreType=cv2.ORB_HARRIS_SCORE )
    '''
    KeyPoint - public attr.:
        pt -> x,y
        size -> affected diameter of the point
    '''
    key_points = orb.detect( img, None ) # find the keypoints with ORB

    ''' compute the descriptors with ORB
    Keypoints for which a descriptor cannot be computed are removed this time. Sometimes new keypoints can be added.
    '''
    key_points, des = orb.compute( img, key_points )
    print( 'orb len: ', len(key_points) )

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints( img, key_points, None, color=(0,255,0), flags=0 )
    imshow_opencv(img2)
    return key_points


def fast_detector( img ):
    #https://docs.opencv.org/4.2.0/df/d0c/tutorial_py_fast.html
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    # fast.setNonmaxSuppression(0) # with this -> 5283; without this -> 1175 for wall-e
    # find and draw the keypoints
    key_points = fast.detect(img,None)
    print( 'fast len: ', len(key_points) )
    img2 = cv2.drawKeypoints(img, key_points, None, color=(255,0,0))
    imshow_opencv(img2)

''' Firstly finds the features in an image by calling 'detect_features', then paints them onto a copy of the same image.
@Params:
* img: Gray-scale image.

@Returns:
* img1: Features on painted (to a copy of the)image.
'''
def paint_features( img, max_feat=0 ):
    corners = detect_features( img, max_feat )
    img_grey = img.copy()
    img_colour = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB )
    print( 'corners len: ', len(corners) )
    for i in corners:
        x,y = i.ravel()
        cv2.circle( img_grey, (x,y), 3, 255, cv2.FILLED )
        cv2.circle( img_colour, (x,y), 3, TURQ, cv2.FILLED )
    imshow_opencv(img_colour)
    return img_grey


def make_display_img( img ):
    # Gray-Scale version of the RGB image
    gray_img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

    # generic info about image
    height, width = gray_img.shape[0], gray_img.shape[1]
    size = gray_img.size
    half_height = height // 2
    half_width = width // 2
    images = []
    tags = []
    ############################################################################
    images.append( img )
    tags.append( 'Original Image' )
    ############################################################################
    # This image's first half is transparent; it is lower half focused.
    alpha_ch_img = combine_images( \
                        draw_rectangle( gray_img, h=half_height ),
                        gray_img )
    images.append( alpha_ch_img )
    tags.append( 'Lower Half Focused' )
    ############################################################################
    # The above image is used, and then half part is added onto it.
    lower_half_img = crop_img( gray_img, y1=half_height )   # image is cropped to half
    lower_half_img = paint_features( lower_half_img, 50 )   # painted with the features
    # NOTE: lower_half_img above could be SAVEd here as "solely lower half+painted with features".
    lower_half_painted_alpha_img = insert_img( alpha_ch_img, lower_half_img, y1=half_height )
    images.append( lower_half_painted_alpha_img )
    tags.append( 'Features Circled' )
    ############################################################################
    # lower_half_left_img = crop_img( lower_half_img, x2=half_width ) # low img cropped to left part
    # NOTE: lower_half_left_img above could be SAVEd here as "solely lower half left-cropped image".
    lower_half_painted_img = insert_img( gray_img, lower_half_img, y1=half_height )
    # NOTE: lower_half_painted_img above could be SAVEd here as "whole img, but solely half part painted with features".

    low_half = draw_rectangle( lower_half_painted_img, h=half_height ) # upper rectangle drawn
    low_left = draw_rectangle( low_half, y=half_height, w=half_width, h=half_height ) # lower right rectangle drawn
    low_left_alpha_img = combine_images( low_left, lower_half_painted_img )

    images.append( low_left_alpha_img )
    tags.append( 'Lower Half Left' )
    ############################################################################
    # lower_half_right_img = crop_img( lower_half_img, x1=half_width )
    # NOTE: lower_half_right_img above could be SAVEd here as "solely lower half right-cropped image".

    low_right = draw_rectangle( low_half, x=half_width, y=half_height, w=half_width, h=half_height ) # lower left rectangle drawn
    low_right_alpha_img = combine_images( low_right, lower_half_painted_img )

    images.append( low_right_alpha_img )
    tags.append( 'Lower Half Right' )
    ############################################################################
    # group_display( images, tags )

if __name__ == '__main__':
    input_img = cv2.imread( sys.argv[1] )
    rgb_img = cv2.cvtColor( input_img, cv2.COLOR_BGR2RGB )
    gray_img = cv2.cvtColor( rgb_img, cv2.COLOR_RGB2GRAY )
    # make_display_img( rgb_img )
    # paint_features( gray_img )

    orb( gray_img )

    # fast_detector( gray_img )
    # orb seems the best
