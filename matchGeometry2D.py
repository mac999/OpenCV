# Program: caculate similarity between 2D geometry using vector
# Author: Kang Tae Wook. laputa99999@gmail.com
# Date: 2020.1
import cv2
import numpy
import sys, getopt

def readContour(file):
	pts = []
	f = open(file, 'r')
	while True:
		line = f.readline()
		if not line:
			break
		tokens = line.split(',')
		if len(tokens) < 2:
			break;
		x = int(tokens[0])
		y = int(tokens[1])
		z = 0.0
		if len(tokens) >= 3:
			z = int(tokens[2])
		pt = [x, y]
		pts.append(pt)	
	f.close()

	ptsNum = numpy.array(pts)
	# print("read contour = {0}".format(len(ptsNum)))
	return ptsNum

def createContours():
	contours = [numpy.array([[50,1],[1,99],[99,99]], dtype=numpy.int32)
			,numpy.array([[50,20],[20,80],[80,80]], dtype=numpy.int32)
			,numpy.array([[1,1],[1,99],[99,99],[50,50],[99,1]], dtype=numpy.int32)]
	return contours

def scaleNormalize(contours, size):
	xlist = []
	ylist = []
	for con in contours:
		for pt in con:
			x, y = pt
			xlist.append(float(x))
			ylist.append(float(y))
	minx = min(xlist)
	maxx = max(xlist)
	originSizeX = maxx - minx
	scale = float(size - 1) / float(originSizeX)
	for con in contours:
		for pt in con:
			pt[0] = int(pt[0] * scale)
			pt[1] = int(pt[1] * scale)
	# print(contours)

def getFiles(argv):
	files = ["point1.csv", "point2.csv", "point3.csv"]
	if len(argv) > 2:
		files.clear
		for f in argv:
			files.append(f)
		del files[0]
	return files

def main(argv):
	files = getFiles(argv)
	
	# create contour and calc match test [[[x y]]]
	contours = []
	# contours = createContours()
	for file in files:
		ct = readContour(file)
		contours.append(ct)
	scaleNormalize(contours, 100)
	# print(contours)

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

if __name__ == "__main__":
	main(sys.argv)