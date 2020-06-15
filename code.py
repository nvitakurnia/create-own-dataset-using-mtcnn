
import cv2
import os
import glob
from PIL import Image
from mtcnn import MTCNN
import tensorflow as tf
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


detector = MTCNN()

imagePath = "/your/image/path/"
data_path = os.path.join(imagePath,'*jpg') #or jpeg,png,etc
files = sorted(glob.glob(data_path)) #for sorted files

for f in files :
    head, tail = os.path.split(f)
    tailname = os.path.splitext(tail)[0]
    image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if not result:
        continue
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    #crop image, this point got from mtcnn 
    frame = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                bounding_box[0]:bounding_box[0]+bounding_box[2]]
    
    #shows the image size
    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]
    print("%s x %s"% (width, height))
    
    if width<64 or height<64 :
        continue
    image_r = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/your/output/folder/%s" % (tail), image_r)
    print("%s image saved" % tailname)
    #print(bounding_box)  
print(result)
