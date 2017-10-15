# Finding-Tennis-Ball
Finding a tennis ball in a picture using a combination of techniques: machine learning (tensorflow), and Region of Image selection, for finding potential matches. It's way slower than the object-detection api provided by tensorflow: https://github.com/tensorflow/models/tree/master/research/object_detection


# Requirements:

Tensorflow: https://www.tensorflow.org/install/

Opencv: https://opencv.org/releases.html

Python 3.6: https://www.python.org/

# Process

I couldn't upload retrained_graph.pb since it exceeded the 25 MB limit on Github upload. 
However, you can make one yourself by retraining Inception's final layer. Google has a nice, and easy to follow Codelab:

Tensorflow for poets: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

To get the pictures for the positive images, you can just take a video and then separate the frames using separate.py, and changing the video filename.

Go through the codelab, and get the retrained_graph.pb and retrained_labels.txt file.

Change the imageFilename, labelsFilename, and graphFilename variables at the beginning of main.py.
Run it on your selected image, wait, and Voila!

# Extreme Latency

main.py works like this:

Blur image

Find contours

Foreach contour, select ROI 100 x 100 pixels centered at the contour, then run the graph on it

If graph result for image is > 90% for tennis ball, then save it, and show that ROI.

This process takes different amounts of time based on the number of contours in your image, the number of images you trained you graph on, etc.

I trained the graph on 150 images of tennis balls, and 60 miscellaneous images

# Notes

The program finds whatever you trained the graph on
