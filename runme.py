import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from moviepy.editor import VideoFileClip


def grayscale(img):
    """
    Returns a gray scale version of the image, obtained by stripping the red and green channel,
    and keeping the blue channel only. Blue channel allows detection of a yellow lane mark in
    bright light. Input images must be in an RGB color space.
    """
    return cv2.split(img)[2]


def region_of_interest(img, vertices):
    """
    Applies an image mask. Only keeps the region of the image defined by the polygon
    formed from the given vertices. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of a Hough transform
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def line_slope(x0, y0, x1, y1):
    """
    Returns the slope of the line going through points (x0, y0) and (x1, y1), i.e. the m coefficient in y=m*x+q;
    but if x0==x1 , i.e. the line is parallel to the y axis, returns inf
    """
    slope = float('inf') if x0 == x1 else (y0 - y1) / (x0 - x1)
    return slope


def line_intercept(x0, y0, x1, y1):
    """
    Returns the intercept of the line going through points (x0, y0) and (x1, y1), i.e. the q coefficient in y=m*x+q;
    but if x0==x1 , i.e. the line is parallel to the y axis, returns inf
    """
    intercept = float('inf') if x0 == x1 else (x1 * y0 - x0 * y1) / (x1 - x0)
    return intercept


def get_Hough_segments(image, vertices, kernel_size, threshold1, threshold2, rho, theta, min_votes, min_line_length,
                       max_line_gap):
    """
    Detect segments within an area of interest in an RGB image. The method first converts the image to grayscale,
    then applies a Gaussian blur, runs a Canny edge detection, discards detected edges that are outside the mask,
    and then runs a probabilistic Hough transform to fetch the list of segments.

    :param image: the input image, to be processed
    :param vertices: the mask, a list of its vertices
    :param kernel_size: kernel size for the Gaussian blur
    :param threshold1: lower threshold for Canny edge detection
    :param threshold2: upper threshold for Canny edge detection
    :param rho: distance resolution in pixels of the Hough grid
    :param theta: angular resolution in radians of the Hough grid
    :param min_votes: minimum number of votes for a line to be considered by the Hough transform
    :param min_line_length: minimum length in pixels of segments
    :param max_line_gap: maximum allowed gaps in pixels between connectable line segments
    :return: the list of detected segments, formatted as [segment0, segment1, ...], where each segment is formatted as
    array([x0, y0, x1, y1], dtype=int32), with x0, y0, x1 and y1 the coordinates of the segment endpoints
    """

    # Go grayscale
    gscale_image = grayscale(image)

    # Do the Gaussian blur
    blur_gray = cv2.GaussianBlur(gscale_image, (kernel_size, kernel_size), 0)

    # Run a Canny edge detection
    edges = cv2.Canny(blur_gray, threshold1, threshold2)

    # Apply a bitmask to set to black all pixels outside the are of interest.
    masked = region_of_interest(edges, vertices)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 720  # angular resolution in radians of the Hough grid
    min_votes = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    segments = cv2.HoughLinesP(masked,
                               rho,
                               theta,
                               min_votes,
                               np.array([]),
                               minLineLength=min_line_length,
                               maxLineGap=max_line_gap)
    # cv2.HoughLinesP() returns a list like [[[segment1]],[[segment2]],...], so unpack the segments and make a list like
    # [[segment1],[segment2],...]
    segments = [segment[0] for segment in segments]

    return segments


def get_lanes(segments, vertices, min_slope):
    """
    Identifies the left and right lane markings based on segments obtained by a Hough transform.
    Segments with a slope below `min_slope` are ignored. The returned lane markings are extended/trimmed vertically
    based on the `vertices` mask, used to mask the input to the Hough transform.
    Endpoint of the left and right lane markings are returned as a list of two tuples:
    [(left_x0, left_y0, left_x1, left_y1), (right_x0, right_y0, right_x1, right_y1)]

    """

    ''' To fit a continuous line on the left, interpolate all endpoints of identified segments with a negative
    slope; to fit a continuous line on the right, do the same with endpoints of segments with a positive
    slope. Slope is the m parameter in the 2D line equation y=m*x+q.'''

    # First some conversion in data types, such that they can be fed to Scikit estimators
    LEFT, RIGHT = 0, 1
    segment_endpoints_x = [[], []]  # The list will accumulate the x coordinate of all identified segment endpoints,
    # partitioned into two sub-lists, one for points belonging to the left side, and one for the right side
    segment_endpoints_y = [[], []]  # Same as above, but for the y coordinates
    for segment in segments:
        slope = line_slope(segment[0], segment[1], segment[2], segment[3])
        # On the pavement there can be nearly horizontal segments (i.e. with slope close to 0) that
        # don't belong to lane marks and would throw off the interpolation, and are therefore discarded
        if abs(slope) < min_slope:
            continue
        side = LEFT if slope < 0 else RIGHT
        segment_endpoints_x[side].append([segment[0]])
        segment_endpoints_x[side].append([segment[2]])
        segment_endpoints_y[side].append(segment[1])
        segment_endpoints_y[side].append(segment[3])

    # Go ahead and interpolate the points with two lines, one for the left side and one of the right side

    def get_x_from_y(y, m, q):
        """
        Returns x such that y=m*x+q if it exists, otherwise return the largest Python floating point number
        """
        return (y - q) / m if m != 0 else sys.float_info.max

    lanes = []  # Initialize the return value
    for side in (LEFT, RIGHT):
        # If no left/right line was detected, then just skip ahead; i.e. don't try to fit and draw it in this image
        if len(segment_endpoints_x[side]) == 0:
            continue
        # Set the random number generator seed used by RANSAC, to make experiments reproducible
        estimator = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), min_samples=2,
                                                 random_state=2111970)
        # The Scikit pipeline is not strictly necessary for least means-square linear interpolation, but it makes it
        # easier to replace the estimator used, and change the degree of the polynomial approximation
        model = make_pipeline(PolynomialFeatures(1), estimator)
        model.fit(segment_endpoints_x[side], segment_endpoints_y[side])
        # The segment to be drawn over the line shall extend (in height) from y=top_y to y=bottom_y
        top_y = min(vertices[1][1], vertices[2][1])
        bottom_y = max(vertices[0][1], vertices[3][1])
        # Fetch the coordinates of two points (x0, y0) and (x1, y1) that belong to the lane line
        x0, x1 = (vertices[0][0], vertices[1][0]) if side == LEFT else (vertices[3][0], vertices[2][0])
        y0 = model.predict(x0)[0]
        y1 = model.predict(x1)[0]
        # Take a segment from the lane line with y coordinate ranging between top_y and bottom_y (i.e. clip/extend
        # the segment (x0, y0)-(x1, y1) as necessary)
        m = line_slope(x0, y0, x1, y1)
        q = line_intercept(x0, y0, x1, y1)
        top_x = int(round(get_x_from_y(top_y, m, q)))
        bottom_x = int(round(get_x_from_y(bottom_y, m, q)))
        lanes.append((bottom_x, bottom_y, top_x, top_y))

    return lanes


def get_vertices(image):
    """
    Returns a list of vertices for the mask, suitable to identify lanes in the image
    """
    height, width, _ = image.shape
    chop_top_w = .42  # The distance of the 2nd vertex of the mask from the left border of the image,
    # and of the 3rd vertex from the right border, expressed as a fraction of the horizontal image size
    chop_top_h = .61  # The distance of the 2nd and 3rd vertex of the mask from the top border of the image,
    # expressed as a fraction of the vertical image size
    chop_bottom_w = .11  # The distance of the 1st and 4th vertex of the mask from the left and right image border
    # respectively
    vertices = np.array([[(width * chop_bottom_w, height), (width * chop_top_w, height * chop_top_h),
                          (width - width * chop_top_w, height * chop_top_h), (width * (1 - chop_bottom_w), height)]],
                        dtype=np.int32)
    return vertices[0]


def draw_segments(image, segments, color, thickness):
    """
    Draws the listed segments with the given RGB color and thickness (in pixels) over an image with the same
    shape as `image`, and return it. The image passed as input parameter is used only to determine the shape, and is
    not altered.
    """
    segments_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # initialize the return value
    for x0, y0, x1, y1 in segments:
        cv2.line(segments_image, (x0, y0), (x1, y1), color, thickness)
    return segments_image


def get_Hough_params():
    """
    A utility function that returns parameters usable to call get_Hough_segments().
    Multiple calls to that function can easly share the same parameter values, if desired.
    """
    return {'kernel_size': 3,  # For the Gaussian blur
            'threshold1': 100,  # Lowest threshold for Canny edge detection
            'threshold2': 200,  # Highest threshold for Canny edge detection
            'rho': 1,  # Distance resolution for Hough transform (pixels)
            'theta': np.pi / 720,  # Angular resolution for Hough transform (radians)
            'min_votes': 25,  # Min number of votes for Hough transoform
            'min_line_length': 10,  # Min segment length (pixels) for Hough transform
            'max_line_gap': 50}  # Max allowed distance (pixels) between connectable segments for Hough transform


def process_lanes_detection(image):
    """
    Return an image same as `image` but with the lane markings (if detected) drawn over it
    """
    vertices = get_vertices(image)
    segments = get_Hough_segments(image, vertices, **get_Hough_params())
    lanes = get_lanes(segments, vertices, 0.3)
    image_with_lanes = draw_segments(image, lanes, [255, 0, 0], 5)
    output_image = weighted_img(image_with_lanes, image)
    return output_image


def process_segments_detection(image):
    """
    Return an image same as `image` but with segments candidate to be edges of lane markings drawn over it
    """
    vertices = get_vertices(image)
    segments = get_Hough_segments(image, vertices, **get_Hough_params())
    image_with_segments = draw_segments(image, segments, [255, 0, 0], 2)
    output_image = weighted_img(image_with_segments, image)
    # To help debugging the edge detection step, uncomment the line below
    # output_image = cv2.cvtColor(masked,cv2.COLOR_GRAY2BGR)
    return output_image


def main():
    plt.interactive(False)  # For debugging within PyCharm

    def process_test_images():
        """
        Process the individual JPEG test images found in ./test_images directory (detect lane markings),
        and write the output as PNG files in the same directory
        """
        input_fnames = os.listdir('test_images')
        input_fnames = [fname for fname in input_fnames if fname.endswith('.jpg')]

        for fname in input_fnames:
            # load the image
            image = mpimg.imread('test_images/' + fname)
            # do your thing
            processed_image = process_lanes_detection(image)
            # save the image
            ext_stripped, _ = os.path.splitext(fname)
            mpimg.imsave('test_images/' + ext_stripped + '.png', processed_image, format='png')

    input_fnames = os.listdir('test_clips')
    input_fnames = [fname for fname in input_fnames if fname.endswith('.mp4')]

    for fname in input_fnames:
        ext_stripped, _ = os.path.splitext(fname)
        clip = VideoFileClip('test_clips/' + fname)
        output_clip_segments = clip.fl_image(process_segments_detection)
        output_clip_segments.write_videofile('output_clips/' + ext_stripped + '_segments.mp4', audio=False)
        output_clip_segments = clip.fl_image(process_lanes_detection)
        output_clip_segments.write_videofile('output_clips/' + ext_stripped + '_lanes.mp4', audio=False)

if __name__ == "__main__":
    main()
