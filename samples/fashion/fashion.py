import os
import sys
import glob
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib, utils
from mrcnn.model import log

# Path to trained weights file
FASHION_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2020"

###################
# Configurations
###################

class FashionConfig(Config):
    # Give the configuration a recognizable name
    NAME = "fashion"
    
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 1 + 21 #
    
    STEPS_PER_EPOCH = 100
    
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################


# BalloonDataset -> TshirtDataset
class FashionDataset(utils.Dataset):
    def load_fashion(self, dataset_dir, subset):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        class_list = ["top", "blouse", "tshirts", "knitwear", "shirts", "bratop", "hoodie",
                      "jean","pants","skirt","leggings","joggerPants", 
                      "coat", "jacket", "jumper", "padding", "vest", "cardigon", "zipUp",
                      "dress", "jumpsuite"]
        
        #Add classes
        num=1
        for i in class_list:
            self.add_class("fashion",num,i)
            num=num+1
        
        #Train or validation dataset
        assert subset in ["Training", "Validation"]
        dataset_dir = os.path.join(dataset_dir, subset)

        codi = ["클래식","프레피",
        "매니시","톰보이",
        "페미닌","로맨틱","섹시",
        "히피","웨스턴","오리엔탈",
        "모던","소피스트케이티드","아방가르드",
        "컨트리","리조트",
        "젠더리스",
        "스포티",
        "레트로","키치/키덜트","힙합","펑크",
        "밀리터리","스트리트"]
        

        json_dir = os.path.join(dataset_dir,"라벨링 데이터")
        image_dir = os.path.join(dataset_dir,"원천 데이터")
        for j in codi:
            json_dir = os.path.join(dataset_dir,j)
            image_dir = os.path.join(dataset_dir,j)
            labeling_data_list=glob.glob(json_dir+'/라벨링 데이터/*.json')
            for i in labeling_data_list:
                file_path = i
                with open(file_path,'r',encoding='UTF=8') as file:
                    file_data = json.load(file)
                    image_id=file_data["이미지 정보"]["이미지 파일명"]
                    image_path=image_dir
                    image=skimage.io.imread(image_path)
                    height=file_data["이미지 정보"]["이미지 높이"]
                    width=file_data["이미지 정보"]["이미지 너비"]
                    rect=file_data["데이터셋 정보"]["데이터셋 상세설명"]["렉트좌표"]
                    polygons=file_data["데이터셋 정보"]["데이터셋 상세설명"]["폴리곤좌표"]

                    
                    self.add_image(
                        "fashion",
                        image_id=image_id,
                        path=image_path,
                        width=width, height=height,
                        rect=rect,
                        polygons=polygons
                    )
        
    
    def load_mask(self, image_id):
        # polygon을 비트맵 마스크로 바꾸는 부분
        # 마스크 shape는 [height, width, instance_count]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        item=["아우터","하의","원피스","상의"]

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            for r in item:
                q=p[r]
                if q:
                    for n, m in enumerate(q)
                        rr,cc = skimage.draw.polygon(q["X좌표"+str(n+1)], q["Y좌표"+str(n+1)])
                        mask[rr, cc, n] = 1
                        if n==len(q)/2-1:
                            break

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    

    #나중에ㅔ에에에에에에에ㅔ
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fashion":
            return info["fashion"]
        else:
            super(self.__class__).image_reference(self, image_id)


def train(model):
    """Train the model."""
    #Training dataset.
    dataset_train = FashionDataset()
    dataset_train.load_fashion(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FashionDataset()
    dataset_val.load_fashion(args.dataset, "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30, layers="3+")

def detect(model):
    
    

############################################################
#  Training
############################################################
if __name__ == '__main__':
    import argparse
    #Parse command line arguments
    

    #Configuration
    if args.command == "train":
        config = FashionConfig()
    else:
        class InferenceConfig(FashionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    #Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
        
    #Load weights
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    if args.weights.lower() == "path":
        model.load_weights(FASHION_MODEL_WEIGHT, by_name=True)
    else:
        model.load_weights(model.find_last(), by_name=True)
    
    #Train or evaluate
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
    if args.command == "train":
        train()
    else:
        detect()