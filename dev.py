#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
#---------------------------------------------------------------------------------------------

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    #img = np.vstack((img, img, img))
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def initify(arr):
    """
    Type casts each element to int

    Parameters
    ----------
    arr: array_like
        Each element is a number

    Returns
    -------
    List of int type casted elements
    """
    return [int(a) for a in arr]


def get_fit_line_end_points(points, imshape):
    """
    Calculates the endpoints of the line segment

    Parameters
    ----------
    points: array_like
        Contains line segment end points in x1, y1, x2, y2 fashion

    imshape: array_like
            Contains shape of the ndarray image frame

    Returns
    -------
    Coodinates of the two end points in x1, y1, x2, y2 fashion
    """
    # Get line parameters
    x, y = [], []
    for _x, _y in points:
        x.append(_x)
        y.append(_y)
    fit = np.polyfit(x, y, 1)

    # Get valid line segment end points
    masked_vertices = get_mask_vertices(imshape)
    y1 = masked_vertices[0][0][1] # bottom_left_outer
    y2 = masked_vertices[0][1][1] # top_left_outer
    x1 = (y1 - fit[1])//fit[0]
    x2 = (y2 - fit[1])//fit[0]

    return initify([x1, y1, x2, y2])


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Draws lines segments on the input image frame

    Parameters
    ----------
    img: ndarray
        Image frame

    lines: array_like
        Contains each element as a line segment end points in x1, y1, x2, y2 fashion

    color: array_like
        Contains channel intensity of the color as int in R, G, B fashion

    thickenss: int
        Thickness of the line drawn in pixels

    """
    if lines is not None:
        left_line_coo, right_line_coo = [], []
        for line in lines:
            for x1,y1,x2,y2 in line:
                mid_x = (x1 + x2)//2
                mid_y = (y1 + y2)//2
                """
                if mid_x < 0.5*img.shape[1]:
                    left_line_coo.append((mid_x, mid_y))

                if mid_x > 0.5*img.shape[1]:
                    right_line_coo.append((mid_x, mid_y))
                """
                if mid_x < 0.5*img.shape[1]:
                    left_line_coo.append((x1, y1))
                    left_line_coo.append((x2, y2))

                if mid_x > 0.5*img.shape[1]:
                    right_line_coo.append((x1, y1))
                    right_line_coo.append((x2, y2))

                #cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2), 10, [0, 255, 0], 2)

        left_x1, left_y1, left_x2, left_y2 = get_fit_line_end_points(left_line_coo, img.shape)
        right_x1, right_y1, right_x2, right_y2 = get_fit_line_end_points(right_line_coo, img.shape)

        # Draw left
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, 10)
        # Draw right
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, 10)
        # Draw divider
        #cv2.line(img, (int(0.5*img.shape[1]), 0), (int(0.5*img.shape[1]), img.shape[0]), [0, 0, 255], 2)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

#---------------------------------------------------------------------------------------------

import os
kernel_size = 5
low_threshold = 50
high_threshold = 100

rho = 2
theta = 2*(np.pi/180)
threshold = 5
min_line_len = 7
max_line_gap = 7

TEST_IMG_DIR = "test_images"
OUTPUT_TEST_IMG_DIR = "output_test_images"

from moviepy.editor import VideoFileClip
#from IPython.display import HTML

mask_bot_left =  (0.12, 1)
mask_top_left = (0.46, 0.6)
mask_top_right = (0.56, 0.6)
mask_bot_right = (0.95, 1)

side_margin = 0.15
top_margin = 0.05
top_inner_width =  0.05
mask_bot_left_inner =  (mask_bot_left[0] + side_margin, mask_bot_left[1])
mask_top_left_inner = (mask_top_left[0] + top_inner_width, mask_top_left[1] + top_margin)
mask_top_right_inner = (mask_top_right[0] - top_inner_width, mask_top_right[1] + top_margin)
mask_bot_right_inner = (mask_bot_right[0] - side_margin, mask_bot_right[1])

def get_mask_vertices(imshape):
    """
    Return double mask image scaled accoriding to the shape of the image frame

    Return an array of two polygons, quadrilaterals specifically, scaled according to the size of
    the image
    """
    vertices = np.array([[(imshape[1]*mask_bot_left[0], imshape[0]*mask_bot_left[1]),
                          (imshape[1]*mask_top_left[0], imshape[0]*mask_top_left[1]),
                          (imshape[1]*mask_top_right[0], imshape[0]*mask_top_right[1]),
                          (imshape[1]*mask_bot_right[0], imshape[0]*mask_bot_right[1])],
                         [(imshape[1]*mask_bot_left_inner[0], imshape[0]*mask_bot_left_inner[1]),
                          (imshape[1]*mask_top_left_inner[0], imshape[0]*mask_top_left_inner[1]),
                          (imshape[1]*mask_top_right_inner[0], imshape[0]*mask_top_right_inner[1]),
                          (imshape[1]*mask_bot_right_inner[0], imshape[0]*mask_bot_right_inner[1])]], dtype = np.int32)

    return vertices

def process_image(img):
    # get image specs
    imshape = img.shape

    # Convert to grayscale
    img_gray = grayscale(img = img)

    # Remove noise, perform gaussian smoothing
    blur_gray = gaussian_blur(img = img_gray,
                              kernel_size = kernel_size)
    # Run canny edge detector
    edges = canny(img = blur_gray,
                  low_threshold = low_threshold,
                  high_threshold = high_threshold)

    # Get region of interest
    vertices = get_mask_vertices(imshape)
    masked_edges = region_of_interest(img = edges, vertices = vertices)
    #cv2.imshow('Title', masked_edges)
    #cv2.waitKey(0)
    #mpimg.imsave(os.path.join(OUTPUT_TEST_IMG_DIR, "output_edges_{}.jpg".format(img_file)), masked_edges)
    # Get Hough lines
    line_image = hough_lines(img = masked_edges,
                             rho = rho,
                             theta = theta,
                             threshold = threshold,
                             min_line_len = min_line_len,
                             max_line_gap = max_line_gap)

    # Relay on original image
    assert(img.shape ==  line_image.shape),"image sizes do no match {} {}".format(img.shape, line_image.shape)
    img_relay = weighted_img(img = line_image, initial_img = img)

    return img_relay


if sys.argv[1] == '1':
    img_file_list = os.listdir(TEST_IMG_DIR)
    for img_file in img_file_list:
        # Read image
        img = mpimg.imread(os.path.join(TEST_IMG_DIR, img_file))
        processed_img = process_image(img)
        mpimg.imsave(os.path.join(OUTPUT_TEST_IMG_DIR, "output_{}".format(img_file)), processed_img)

    print("TESTS COMPLETED")

if sys.argv[1] == '2':
    TEST_VID_DIR = "test_videos"
    #video_filename = "solidWhiteRight.mp4"
    video_filename = "solidYellowLeft.mp4"
    clip1 = VideoFileClip(os.path.join(TEST_VID_DIR, video_filename))

    white_output = os.path.join(TEST_VID_DIR, "output_{}".format(video_filename))
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

