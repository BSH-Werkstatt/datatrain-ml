# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

from imageio import imread

# define the test configuration
class TestConfig(Config):
     NAME = "washingmachine"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
	 def __init__(self, class_names):
     	self.NUM_CLASSES = len(class_names)
	 
def define_model(model = None, class_names = None):
	K.clear_session()

	if model is None:
		model = 'mask_rcnn_coco.h5'
	
	if class_names is None:
		# define 81 classes that the coco model knowns about
		class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
				'bus', 'train', 'truck', 'boat', 'traffic light',
				'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
				'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
				'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
				'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
				'kite', 'baseball bat', 'baseball glove', 'skateboard',
				'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
				'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
				'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
				'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
				'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
				'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
				'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
				'teddy bear', 'hair drier', 'toothbrush']

	rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig(class_names))
	rcnn.load_weights(model, by_name=True)

	return rcnn, class_names

def predict_img(filename, rcnn):
	# load photograph
	#img = load_img('images/banana_apple.jpg')
	#img = load_img('images/' + filename)
	img = imread(filename)
	img = img_to_array(img)
	# make prediction
	results = rcnn.detect([img], verbose=0)
	# get dictionary for first prediction
	r = results[0]
	return r, img

def visualize(r, img, class_names, filename):	
	# show photo with bounding boxes, masks, class labels and scores
	display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], making_image=True, detect=False, prediction_image_filename=filename) # making_image saves image
	return