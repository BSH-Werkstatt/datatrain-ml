### DEMO ###
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

import model_setup as m

rcnn, class_names = m.define_model()

filename = input('Select picture to predict: ')
while filename != '':
	prediction, image = m.predict_img(filename, rcnn)
	m.visualize(prediction, image, class_names)
	print(prediction)
	filename = input('\nSelect picture to predict: ')