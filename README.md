# Repository Overview 
This repository contains Python code for self-driving cars that use computer vision and deep learning techniques to address problems that autonomous vehicles face, such as detecting lane lines and predicting steering angles in real-time.


## Project Setup

### OpenCV
![OpenCV](screenshots/opencv.png)

OpenCV stands for “Open Source Computer Vision” is a library for computer vision and machine learning software library. OpenCV has C++, Python, Java and MATLAB interfaces and supports Windows, Linux, Android, and Mac OS. 

## Use OpenCV to Load Image 
The cv2.imread() method loads an image from the specified file. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.

```python
image = cv2.imread('/data/Lane_Original.jpg')
```
### Original Image
![Original-Image](screenshots/Lane_Original.jpg)
In a real-life scenario, an autonomous vehicle would process a video instead of an image, but for the sake of learning, we demonstrate how to detect the lanes in a particular image first. 

## Apply Edge Detection Algorihtm to Image
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images

```python
def detect_edges(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    blurred_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)  
    return edges   
```

# STEP 3: USE LANE EDGES TO DEFINE REGION OF INTEREST #
# STEP 4: ISOLATE REGION OF INTEREST IN THE IMAGE
# STEP 5: USE HOUGH TRANSFORM TO FIND LANE LINES 
# STEP 6: OPTIMIZE LANE LINES 

## Download Road Image
https://github.com/rslim087a/road-image

## Install OpenCV Libary in Terminal:
$ pip install opencv-contrib-python



