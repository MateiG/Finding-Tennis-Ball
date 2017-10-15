import cv2
import numpy as np
import tensorflow as tf
import os


imageFilename = 'tennisball.png'
labelsFilename = 'retrained_labels.txt'
graphFilename = 'retrained_graph.pb'
img = cv2.imread(imageFilename)
kernel = np.ones((5,5),np.float32)/25

def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

labels = load_labels(labelsFilename)
load_graph(graphFilename)

human_string = ''
score = 0
potential_matches = []
timer = 1

def run_graph(image_data, labels, input_layer_name, output_layer_name, num_top_predictions):
	global human_string
	global score
	global potential_matches
	global timer

	with tf.Session() as sess:
	    # Feed the image_data as input to the graph.
	    #   predictions  will contain a two-dimensional array, where one
	    #   dimension represents the input image count, and the other has
	    #   predictions per class
	    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
	    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

	    # Sort to show labels in order of confidence
	    top_k = predictions.argsort()[-num_top_predictions:][::-1]
	    
	    for node_id in top_k:
	      human_string = labels[node_id]
	      score = predictions[node_id]
	      if human_string == 'ball':
	      	if score >= 0.9:
	      		potential_matches.append(timer)
	      		potential_matches.append(score)

	    return 0


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

contour_list = []
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.filter2D(gray,-1,kernel)
edges = auto_canny(blurred)
_, contours, _= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    if ((len(approx) > 1) & (area > 15)):
        contour_list.append(contour)

coordinates_of_roi = []

for cnt in contour_list:

	M = cv2.moments(cnt)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	roi = img[cY - 50:cY + 50, cX - 50:cX + 50]
	roiFilename = 'C:/Matei/Python/TENNISBALL/MOREADVANCED/roi' + str(timer) + '.png'
	coordinates_of_roi.append([cX,cY])
	cv2.imwrite(roiFilename, roi)
	image_data = load_image(roiFilename)
	timer += 1
	try:
		run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)
	except:
		print("No image")   
		print(potential_matches)
		break
def identify():
	scores_for_class = potential_matches[1::2]
	highest = max(scores_for_class)

	location = potential_matches.index(highest)

	roi_img_number = potential_matches[location - 1]

	correct_roi = cv2.imread('C:/Matei/Python/TENNISBALL/MOREADVANCED/roi' + str(roi_img_number) + '.png')
	print(coordinates_of_roi[roi_img_number - 1])
	cv2.imshow('Found It!', correct_roi)
try:
	identify()
except:
	print("Coudn't identify")

cv2.drawContours(img, contour_list,  -1, (255,0,0), 2)
cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()
