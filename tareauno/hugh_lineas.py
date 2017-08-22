import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#leyendo una imagen
img = cv2.imread('exit_ramp.jpg')
image = img
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 0)
#cv2.imshow('gaussian', gray)

#aplicando canny
edges = cv2.Canny(gray,10,250,apertureSize = 3)
cv2.imshow('Canny', edges)

cv2.imshow('gray',gray)
rows, cols = edges.shape[:2] # dimension de la imagen
print(rows,cols)

#Defining each edges
abajo_left  = [cols*0.1, rows*0.95]
superior_left     = [cols*0.49, rows*0.52]
abajo_right = [cols*0.95, rows*0.95]
superior_right    = [cols*0.5, rows*0.52]

abajo_right_int = [cols*0.7, rows*0.95]
superior_right_int = [cols*0.5, rows*0.6]
abajo_left_int = [cols*0.3, rows*0.95]
superior_left_int = [cols*0.5, rows*0.6]

vertices = np.array([[abajo_left, superior_left, superior_right, abajo_right, abajo_right_int, superior_right_int, superior_left_int, abajo_left_int]], dtype=np.int32)
mask = np.zeros_like(edges)
#print ("mask", mask)

if len(mask.shape)==2:
	poli = cv2.fillPoly(mask, vertices, 255)
else:
	cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel d
cv2.imshow('poligono', poli)

select_region = cv2.bitwise_and(edges, mask)
cv2.imshow('region',select_region)

minLineLength = 10 #10
maxLineGap = 20	#20
# se sacan lineas unicamente de la region de interes
lines = cv2.HoughLinesP(select_region,1,np.pi/180,80,minLineLength,maxLineGap) #np.pi/180
#print("lines",lines)

left_lines = []
left_weights = []
right_lines = []
right_weights = []

lim_inf_slope = 0.2
lim_sup_slope = 0.8
contador_izq = 0
contador_der = 0
slope_izq = 0
slope_der = 0
inter_izq = 0
inter_der = 0

for x in range (0,len(lines)):
	for x1, y1, x2, y2 in lines[x]:
		slope = float(y2 -y1) / (x2 -x1)
		if (x1 == x2 or y1 == y2):
			#print ('fuera de rango',slope)
			continue
		elif (slope > lim_inf_slope and slope < lim_sup_slope) or (slope < -lim_inf_slope and slope > -lim_sup_slope):
			#contador = contador +1
			cv2.line(gray,(x1,y1),(x2,y2),(0,0,0),2)
			#print('en rango',slope)
			intercept = y1 - slope*x1
			length = np.sqrt((y2-y1)**2+(x2-x1)**2)
			if slope < 0:
				contador_izq = contador_izq +1
				slope_izq = slope_izq + slope
				inter_izq = inter_izq + intercept 
				left_lines.append((slope, intercept))
				left_weights.append((length))
			else:
				contador_der = contador_der +1
				slope_der = slope_der+slope
				inter_der = inter_der + intercept
				right_lines.append((slope, intercept))
				right_weights.append((length))
		
left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
slope_izq = slope_izq/contador_izq
slope_der = slope_der/contador_der
inter_izq = inter_izq/contador_izq
inter_der = inter_der/contador_der	



y1 = gray.shape[0]
y2 = 0.6*y1

#print(left_lane[1])

#x1 = int((y1 - left_lane[1])/left_lane[0])
#x2 = int((y2 - left_lane[1])/left_lane[0])
x1 = int((y1 - inter_izq)/slope_izq)
x2 = int((y2 - inter_izq)/slope_izq)
y1 = int (y1)
y2 = int (y2)
#print('coord izq',x1,y1,x2,y2)

cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)

y1 = gray.shape[0]
y2 = 0.6*y1

#print(right_lane[1])

#x1 = int((y1 - right_lane[1])/right_lane[0])
#x2 = int((y2 - right_lane[1])/right_lane[0])
x1 = int((y1 - inter_der)/slope_der)
x2 = int((y2 - inter_der)/slope_der)
y1 = int (y1)
y2 = int (y2)
#print('coord der',x1,y1,x2,y2)

cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)
    
cv2.imshow('lineas',gray)
cv2.imshow('hough',img)

#cv2.imshow('lineas',gray)

cv2.waitKey(0)

