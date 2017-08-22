import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


imagen = cv2.imread('exit_ramp.jpg')
image = imagen
gray = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
print('This image is ', type(imagen), ' with dimensions', imagen.shape)

#eliminar ruido con gaussian
gray = cv2.GaussianBlur(gray, (3,3), 0)
kernel_size = 9;
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

#usaremos canny

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold,apertureSize = 3)

#agregar una mascara
mask = np.zeros_like(edges)   
ignore_mask_color = 255 
rows, cols = edges.shape[:2] # dimension de la imagen

bottom_left  = [cols*0.1, rows*0.95]
top_left     = [cols*0.49, rows*0.52]
bottom_right = [cols*0.95, rows*0.95]
top_right    = [cols*0.5, rows*0.52]

bottom_right_int = [cols*0.7, rows*0.95]
top_right_int = [cols*0.5, rows*0.6]
bottom_left_int = [cols*0.3, rows*0.95]
top_left_int = [cols*0.5, rows*0.6]

vertices = np.array([[bottom_left, top_left, top_right, bottom_right, bottom_right_int, top_right_int, top_left_int, bottom_left_int]], dtype=np.int32)
mask = np.zeros_like(edges)


if len(mask.shape)==2:
	poli = cv2.fillPoly(mask, vertices, 255)
else:
	cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # En caso de que la entrada tenga un canal d

cv2.imshow('poligono', poli)

select_region = cv2.bitwise_and(edges, mask)
#cv2.imshow('region',select_region)

#plt.imshow(edges)
#plt.show()

slope_threshold = 0.5;

# Define la transformada de hough para lineas largas

# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 70    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
#line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
#remove detected lines from edges

image2show = np.copy(imagen)
for line in lines:
    for x1,y1,x2,y2 in line:
        slope = (y2-y1)/(x2-x1)
        #print(slope)
        if ( slope > slope_threshold or slope < -slope_threshold):
            cv2.line(image2show,(x1,y1),(x2,y2),(0,0,255),3)
            cv2.line(edges,(x1,y1),(x2,y2),(0,0,0),3)

# Define the Hough transform parameters for dotted lines

# Make a blank the same size as our image to draw on
rho = 3 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 50    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100 #minimum number of pixels making up a line
max_line_gap = 50    # maximum gap in pixels between connectable line segments
#line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines_dotted = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines_dotted:
    for x1,y1,x2,y2 in line:
        slope = (y2-y1)/(x2-x1)
        #print(slope)
        if ( slope > slope_threshold or slope < -slope_threshold):
            cv2.line(image2show,(x1,y1),(x2,y2),(0,255,255),3)

# Draw the lines on the edge image
#lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
#plt.imshow(image2show)
#plt.show()
cv2.imshow('poligono', image2show)

cv2.waitKey(0)

