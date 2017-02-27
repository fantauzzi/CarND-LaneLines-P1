#**Finding Lane Lines on the Road**
##*Writeup for Project 1 of Udacity's Self-Driving Car Nanodegree Program*
##**The Pipeline**
*The short of it:*
 - Convert the frame to grayscale by stripping the R and G channel, and keeping the B channel.
 - Apply a Gaussian blur.
 - Run it through Canny edge detection.
 - Apply a mask to blacken pixels outside of the area of interest.
 - Detect segments with probability Hough transform.
 - Discard segments whose slope in absolute value is below a certain threshold.
 - Identify segments as belonging to the left or right lane marking, based on their slope, and interpolate them linearly one side at a time with RANSAC.
 - draw the two linear interpolations over the original image, such that they remain within the area of interest (the area masked after edge detection)

I look for all files with extension .mp4 in the input directory, and process them one at a time. Each file is processed twice, providing two output clips: one that highlights segments identified with a Hough transform, which are candidate to be part of the lane markings, and one that highlights the left and right lane markings. 

For each frame, the first step in its processing is to convert it to grayscale, in order to feed it to a Canny edge detection. There are many ways to do an RGB to grayscale conversion; the OpenCV cv2.cvtColor() function, when converting from RGB, computes a blend of the R, G and B channels. That gave me a hard time detecting yellow markings on a brightly lit pavement, such as in the "challenge.mp4" test clip. Because the contrast is different in each of the three RGB channels, I have tried using each of them, in turn, as a gray scale image, and got the best results with the blue channel. In that channel white markings are still bright, while yellow markings become darker than the pavement, and Canny edge detection easily pick them up. I use cv2.split() to extract the blue channel, and use it as grayscale image.

I then blur the frame with a Gaussian filter to lower the impact of noise, and feed it to a Canny edge detection. Proper tuning of its parameters allows to outline the lane markings, but invariably also picks up a lot of unrelated edges. For this reason, I apply a mask to the output of edge detection, blackening pixels that, because of their position, cannot be part of the markings. The mask is shaped like a trapezium, and will also be used to calculate the correct vertical extensions of the detected lane lines. I therefore calculate the mask in an appropriate function, that I can call as needed: get_vertices().

The masked output of Canny edge detection is then sent to the probabilistic Hough transform. In order to find segments accurately enough I set the angular resolution of the transform to 1/4th of a degree, and the linear resolution to 1 pixel. 

Now I have a list of segments that are candidate to be part of the lane markings. However, some of them are additional markings on the pavement, shadows, changes in pavement color, and in the "challenge.mpg" clips also the hood of the car. To reduce noise, I discard all detected segments that have a slope in absolute value below a certain threshold, i.e. that are "too horizontal".
I identify whether every segment should be part of the left or right lane markings based on their slope, and then interpolate them one per side, to get a line. For linear interpolation I use RANSAC, which has some robustness toward outliers, and gave me better results that a plain least mean-square linear interpolation. The latter was more affected by segments extraneous to the markings, e.g. belonging to bushes on the side of the road and other cars.

From interpolation I get two lines, and then trim them and draw them over the picture. The trimming is done in such a way that the drawing remains within the masked area.
##**Limitations**
In case of a bend tight enough I expect my implementation to miss the lane markings, as the detected segments may be discarded for having a slope too close to 0 in absolute value.

More in general, telling apart segments that belong to lane markings from those that don't is difficult. Thresholding them based on their slope is a crude way to remove outliers, rather effective with extraneous markings on the pavement and the edge of the car hood, but ineffective with everything else, such as shadows, bushes and other objects on the sides of the road, other vehicles. 

The current interpolation of detected segments is actually an interpolation of their end-points, i.e. it doesn't keep into consideration the lines joining them, therefore discarding potentially useful information. 

Rain, snow and darkness, could make the currently adopted edge detection ineffective. Moreover, lane markings could be obstructed by vehicles, and become temporarily undetectable by my program.
##**Possible improvements**
Canny edge detection is based on image tones alone (in fact, it works on a grayscale version of it); an algorithm that takes also the color into consideration might work more reliably, useful in difficult environmental conditions (rain, etc.).

To reliably identify segments that belong to the lane markings, a better approach would be a proper classification algorithm, as opposed to the simple thresholding based on their slope.

It would be interesting to explore a variation of RANSAC that, instead of interpolating one line at a time, on a sub-set of the detected segments, interpolates the two lines at the same time, over the whole set of segments.
