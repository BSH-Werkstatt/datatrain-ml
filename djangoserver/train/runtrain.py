"""
Mask R-CNN
Train on the surgery robot dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    #Train a new model starting from pre-trained COCO weights
    python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=coco

    #Train a new model starting from pre-trained ImageNet weights
    python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=imagenet

    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    python surgery.py train --dataset=/home/.../mask_rcnn/data/surgery/ --weights=last

    #Detect and color splash on a image with the last model you trained.
    #This will find the last trained weights in the model directory.
    python surgery.py splash --weights=last --image=/home/...../*.jpg

    #Detect and color splash on a video with a specific pre-trained weights of yours.
    python sugery.py splash --weights=/home/.../logs/mask_rcnn_surgery_0030.h5  --video=/home/simon/Videos/Center.wmv
"""

import os
import sys
import json
import math
import datetime
import numpy as np
import skimage.draw
import cv2
from matplotlib import pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SurgeryConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "default"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100 #100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self, taxonomy):
        # Number of classes (including background)
        self.NUM_CLASSES = 1 + len(taxonomy)  # Background + objects
        Config.__init__(self) # run __init__ from Config


############################################################
#  Dataset
############################################################

class SurgeryDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, campaignId, taxonomy, annotations, hc=False):
        """Load the surgery dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """
        # Add classes. We have only one class to add.
        for idx, label in enumerate(taxonomy):
            #print('ADD: ', campaignId, ' num ', idx, ' label ', label, '\n')
            self.add_class(campaignId, idx+1, label)
        '''self.add_class("fridges", 1, "Cauliflower")
        self.add_class("fridges", 2, "Lettuce")
        self.add_class("fridges", 3, "Egg carton")
        self.add_class("fridges", 4, "Youghurt")'''
        if hc is True:
            for i in range(1,14):
                self.add_class("surgery", i, "{}".format(i))
            self.add_class("surgery", 14, "arm")

        # Train or validation dataset?
        assert subset in ["train", "val", "predict"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {name:'a'},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        #annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        #annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        image_path = [a['url'] for a in annotations if a['annotations']]
        image_id = [a['id'] for a in annotations if a['annotations']]
        images = [a['annotations'] for a in annotations if a['annotations']] # 'annotations' cannot be []
        # Add images
        for idx, i in enumerate(images):
            names = [{'label': a['label']} for a in i]
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stored in the
            # points (see json format above)
            all_points_x = [math.floor(a['x']) for a in i for a in a['points']]
            all_points_y = [math.floor(a['y']) for a in i for a in a['points']]
            polygons = [{'name': a['type'], 'all_points_x': all_points_x, 'all_points_y': all_points_y} for a in i] #a['regions'].values()
			# load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #image_path = [i["url"] for i in i] #'http://ios19bsh.ase.in.tum.de/dev/api/images/' + campaignId + '/' + image_id[idx] + '.jpg'
            image = skimage.io.imread(image_path[idx])
            height, width = image.shape[:2]

            self.add_image(
                campaignId,
                image_id= image_id[idx] + '.jpg',  # use file name as a unique image id
                path=image_path[idx],
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id, campaignId, taxonomy):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a surgery dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != campaignId: #"surgery"
            return super(self.__class__, self).load_mask(image_id, campaignId, taxonomy)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if p['label'] in taxonomy:
                class_ids[i] = taxonomy.index(p['label']) + 1 # 0 is BG
            '''if p['Food'] == 'Cauliflower':
                class_ids[i] = 1
            elif p['Food'] == 'Lettuce':
                class_ids[i] = 2
            elif p['Food'] == 'Egg carton':
                class_ids[i] = 3
            elif p['Food'] == 'Youghurt':
                class_ids[i] = 4'''
            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "surgery": # name of the project (e.g. surgery)
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask_hc(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a surgery dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "surgery":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
            if p['name'] == 'arm':
                class_ids[i] = 14
            elif p['name'] == 'error':
                pass
            else:
                class_ids[i] = int(p['name'])
            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids

def train(model, dataset, campaignId, taxonomy, annotationsTrain, annotationsVal, config, *dic):
    """Train the model."""

    # Training dataset.
    dataset_train = SurgeryDataset()
    dataset_train.load_VIA(dataset, "train", campaignId, taxonomy, annotationsTrain)
    dataset_train.prepare()
    print('dataset_train: ', dataset_train)
    # Validation dataset
    dataset_val = SurgeryDataset()
    dataset_val.load_VIA(dataset, "val", campaignId, taxonomy, annotationsVal)
    dataset_val.prepare()

    # *** This training schedu le is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it. (Heads are the decision layer neurons)
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                config.LEARNING_RATE, #learning_rate
                5, #60 epochs
                'heads', #layers
                campaignId, 
                taxonomy) 


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, taxonomy, image_path=None, video_path=None, out_dir=''):
    assert image_path or video_path

    class_names = taxonomy #['BG', 'Cauliflower', 'Lettuce', 'Egg carton', 'Youghurt']
    class_names.insert(0, 'BG')

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image))
        # Read image
        image = skimage.io.imread(image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        # splash = color_splash(image, r['masks'])
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], making_image=True)
        file_name = 'splash.png'
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        save_file_name = os.path.join(out_dir, file_name)
        skimage.io.imsave(save_file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 1600
        height = 1600
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        #For video, we wish classes keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))
        while success:
            print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # splash = color_splash(image, r['masks'])

                splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'], colors=colors, making_video=True)
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def detect(model, dataset_dir, subset, campaignId, taxonomy, annotations):
    """Run detection on images in the given directory."""
    print("Detection running on {}".format(dataset_dir))

    os.makedirs('RESULTS')
    submit_dir = os.path.join(os.getcwd(), "RESULTS/")
    # Read dataset
    dataset = SurgeryDataset()
    dataset.load_VIA(dataset_dir, subset, campaignId, taxonomy, annotations)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        canvas = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'], detect=True)
            # show_bbox=False, show_mask=False,
            # title="Predictions",
            # detect=True)
        canvas.print_figure("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"][:-4]))
    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
############################################################
#  Training
############################################################

def train_main(command, weights, campaignId, taxonomy, annotations, dataset=None, image=None, video=None, subset=None, logs=DEFAULT_LOGS_DIR):
    # Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect rings and robot arms.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/home/simon/mask_rcnn/data/surgery",
                        help='Directory of the surgery dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/home/simon/logs/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()
    '''
    # Split annotations json into 2: val and train .json
    annotations = [a for a in annotations if a["annotations"]] # only images with annotations considered
    numImages = len(annotations)
    percentageTrain = 0.8 # percentage of images for training (1 - percentageTrain = percentageVal)
    annotationsTrain = annotations[:math.floor(numImages*percentageTrain)]
    annotationsVal = annotations[math.floor(numImages*percentageTrain):]


    # Validate arguments
    if command == "train":
        assert dataset, "Argument --dataset is required for training"

    elif command == "splash":
        assert image or video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", weights)
    print("Dataset: ", dataset)
    print("Logs: ", logs)

    # Configurations
    if command == "train":
        config = SurgeryConfig(taxonomy)
    else:
        class InferenceConfig(SurgeryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig(taxonomy)
    config.display()

    # Create model
    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if command == "train":
        train(model, dataset, campaignId, taxonomy, annotationsTrain, annotationsVal, config)
    elif command == "detect":
        detect(model, dataset, subset, campaignId, taxonomy, annotations)
    elif command == "splash":
        detect_and_color_splash(model, taxonomy, image_path=image,
                                video_path=video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(command))



if __name__ == '__main__':
    import argparse
# dataset_dir = '/home/simon/deeplearning/mask_rcnn/data'
# dataset_train = SurgeryDataset()
# dataset_train.VIA(dataset_dir, "train")
# # dataset_train.prepare()
# a, b = dataset_train.load_mask(130)
# print(a.shape, b.shape)
# print(b)
