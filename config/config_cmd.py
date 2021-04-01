from datetime import datetime
#prepare json file dataset
image_dir = 'img/tree'
json_dir = 'img/tree.json'


#Train model
#select data
dataformat = ['rehab']#coco_mpii_13,coco_mpii_16,yoga,coco,ai_coco,coco_crowd
datanumber = 1 #multiple
dataset_comment = "yoga"
joint_weight= [1,1,5]
use_different_joint_weights = False

#model type :
# mobilenetv1: inputsize(160,160), ouput size (10,10)
# mobilenetv2: inputsize(224,224), output size (14,14)
# mobilenetv3: inputsize(224,224), output size (7,7)
# hourglass: inputsize(256,256), output size (64,64)
# efficientnet : inputsize(224,224), output size(56,56)
# efficientnet : inputsize(224,224), output size(56,56)

#data prepare
# train_annotFile = "img/ai_challenger/aic_train.json"
# train_imageDir = "img/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902"
# test_annotFile = "img/ai_challenger/aic_val.json"
# test_imageDir = "img/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911"
# train_annotFile = "img/crowdpose/annotations/crowdpose_val.json"
# train_imageDir = "img/crowdpose/images"
# test_annotFile = "img/crowdpose/annotations/crowdpose_val.json"
# test_imageDir = "img/crowdpose/images"
# train_annotFile = "img/ochuman/ochuman_coco_format_test_range_0.00_1.00.json"
# train_imageDir = "img/ochuman/images"
# test_annotFile = "img/ochuman/ochuman_coco_format_test_range_0.00_1.00.json"
# test_imageDir = "img/ochuman/images"
# train_annotFile = "img/mpiitrain.json"
# train_imageDir = ""
# test_annotFile = "img/mpiitrain.json"
# test_imageDir = ""
# train_annotFile = "img/single_yoga2_train_new.json"
# train_imageDir = "img/single_yoga2_train_new"
# test_annotFile = "img/single_yoga2_test.json"
# test_imageDir = "img/single_yoga2_test"
# train_annotFile = "img/ai_add_searchedyoga_train.json"
# train_imageDir = "img/ai_add_searchedyoga_train"
# test_annotFile = "img/ai_add_searchedyoga_test.json"
# test_imageDir = "img/ai_add_searchedyoga_test"
# train_annotFile = "img/yoga_train.json"
# train_imageDir = "img/yoga_train"
test_annotFile = "img/yoga_test.json"
test_imageDir = "img/yoga_test"
train_annotFile = "img/rehab_train.json"
train_imageDir = "img/rehab_train"

dataprovider_trainanno =[train_annotFile]
dataprovider_trainimg = [train_imageDir]
dataprovider_testanno = [test_annotFile]
dataprovider_testimg = [test_imageDir]
inputSize = (224,224)
inputshape = (None, 224, 224, 3)
outputSize = (56,56)
modelchannel = [1,1,1,1,1,1]
#8[0.85,0.85,0.85,0.85,0.85,0.85]
#7[0.75,0.75,0.75,0.75,0.75,0.75]
#6[1,0.25,0.25,0.25,0.25,1]
#5[1,0.5,0.5,0.5,0.5,1]
#4[1,0.75,0.75,0.75,0.75,1]
#3[1,1,0.5,0.5,1,1]
#2[1,1,0.25,0.25,0.25,1]
#1[1,1,0.75,0.75,0.75,1]
#0[1,1,0.5,0.5,0.5,1]

inputshapeforflops = [(1, 224, 224, 3)]

#For DUC
pixelshuffle = [2]
convb_13 = [[3,3,52,1,"psconv1"]]
convb_16 = [[3,3,64,1,"psconv1"]] #64

#comment :images number
total_images = 200000
dataset = 'senet, block=16'
