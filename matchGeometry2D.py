# Program: caculate similarity between 2D geometry using vector
import cv2
import numpy

def createContours():
	contours = [numpy.array([[50,1],[1,99],[99,99]], dtype=numpy.int32)
			,numpy.array([[50,20],[20,80],[80,80]], dtype=numpy.int32)
			,numpy.array([[1,1],[1,99],[99,99],[50,50],[99,1]], dtype=numpy.int32)]
	return contours

# create contour and calc match test [[[x y]]]
contours = createContours()

drawing = numpy.zeros([100, 100], numpy.uint8)
drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)
index = 0
for cnt in contours:
	rgb = (0, 70 * (index + 1), 0)
	cv2.drawContours(drawing, [cnt], 0, rgb, 2)
	index = index + 1
cv2.imshow('output',drawing)

print("2D Geometry Distances Between \n-------------------------")
index = 0
for contour in contours:
	m = cv2.matchShapes(contours[0], contour, cv2.CONTOURS_MATCH_I1, 0) # 1, 0)  # enum { CV_CONTOURS_MATCH_I1  =1, CV_CONTOURS_MATCH_I2  =2, CV_CONTOURS_MATCH_I3  =3};
	print("{0} and {1} : {2}".format(0, index, m))
	h = cv2.HuMoments(cv2.moments(contour)).flatten()		# https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/
	print(h)
	index = index + 1
cv2.waitKey(0)
