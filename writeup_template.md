# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./rnd/double_mask/mask.jpg "Double Mask"
[image3]: ./rnd/line_mid_points/output_solidWhiteCurve.jpg "line separation"
[image4]: ./output_test_images/output_solidWhiteCurve.jpg "line extrapolation"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .

#### Steps: 5

#### Preprocessing
1. Conversion to gray scale
   ![alt text][image1]

2. Removing Noise --> Gaussian blur

#### Detect edges
3. Canny Edge detection

4. Gets region of interest
   I masked/bounded the lane region with 8 vertice polygon.
   That's how the mask look like.

   ![alt text][image2]

5. Ran Hough transform on the masked image to find straight lines in the region of interest


#### Extrapolate lines
6. Modified the draw_lines() routine in the following way:
   1. Assumed the center to be at 50% of the image width  
   ![alt text][image3]
   2. Separated detected hough lines end-points, on the basis of whether the mid point of the line lies on left 50% or right 50% of the image, into respective left lane and right lane bucket. 
   3. Performed a linear fit on the all the points in either in bucket, resulting in a single line.
   4. This step involves getting line segment end-points, using above line and region of interest boundary, to be drawn.
   ![alt text][image4]
        

### 2. Identify potential shortcomings with your current pipeline


1. As the mask is super hard coded, it won't sustain at turns.

2. Camera movement disturbance will offshoot mask.

3. Changing lanes would again misplace the mask. 


### 3. Suggest possible improvements to your pipeline

1. A possible scenario would be to create a 2-D histogram of slope-intercept of all the lines detected by Hough Transform. We should see two peaks regions given sufficient bin size. I tried this approach but for some reason I couldn't get through the 2-D histogram

2. Using linear fit approach is not mathematicaly sound as it minimized vertical distance of point to line. Instead it would make more sense to minimize perpendicular distance of points from hypothesis line.`
