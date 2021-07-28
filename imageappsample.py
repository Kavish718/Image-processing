import cv2
import numpy as np
from PIL import Image

warped_img = None
scanList=[]
for i in range(6):
	input_img = cv2.imread("images\doc{}.jpg".format(i))
	input_img = cv2.resize(input_img, (480, 640))
	rect = np.zeros((4, 2), dtype="float32")
	img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(img, (5, 5), 2)
	threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	img = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
	gray = cv2.bilateralFilter(img, 7, 17, 17)
	edged = cv2.Canny(gray, 120, 255, apertureSize=5)
	cv2.imshow("canny",edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print(contours)
	cnts = sorted(contours, key=cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			screen = approx
			break
	pts = screen.reshape(4, 2)
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(input_img, M, (maxWidth, maxHeight))
	warped_img = cv2.resize(warped, (480, 640))
	#kernel for sharpening
	kernel = np.array([[0, -1,0], [-1, 5, -1], [0, -1,0]])
	warped_img = cv2.filter2D(warped_img, -1, kernel)
	hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
	value = 30
	h, s, v = cv2.split(hsv)
	lim = 255 - value
	v[v > lim] = 255
	v[v <= lim] += value
	s[s > lim] = 255
	s[s <= lim] += value
	final_hsv = cv2.merge((h, s, v))
	warped_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
	resultimage = np.zeros((480, 640))
	# normalizing the given image using normalize() function
	warped_img = cv2.normalize(warped_img, resultimage, 0, 255, cv2.NORM_MINMAX)
	scanList.append(warped_img)

imageList=[]
for i in scanList:
	img = Image.fromarray(i)
	imageList.append(img)
img1 = imageList[0]
img1.save(r'Scan.pdf',save_all=True, append_images=imageList[1:])
print('PDF file saved !')