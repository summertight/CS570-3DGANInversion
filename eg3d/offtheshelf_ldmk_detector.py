import imageio as io
import torch, os
import numpy as np
from skimage.transform import resize
from facenet_pytorch import MTCNN, InceptionResnetV1
#import sys
#sys.path.append('./training.facenet_pytorch_folder')
#from training.facenet_pytorch_folder.model.mtcnn import MTCNN
import glob
# If required, create a face detection pipeline using MTCNN:

import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo


class LandmarkModel():
    def __init__(self, name, root='./checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        return kps

    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss
mtcnn = MTCNN(image_size=(256), margin=0)
path = './inthewild_datav2'
path = './sample_refined'
path = '/home/nas4_user/jaeseonglee/4D-Facial-Avatars/nerface_dataset/person_2/train/crop'
path = '/home/nas4_user/jaeseonglee/dataset/ffhq256'
path = '/home/nas1_userB/dataset/Celeb_HQ/val'
landmarkModel = LandmarkModel(name='landmarks', root = '/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts')
landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(256,256))
import imageio as io
from skimage.transform import resize
save_path =path+'_ldmks'
os.makedirs(save_path, exist_ok=True)

os.makedirs(save_path+'/male', exist_ok=True)
os.makedirs(save_path+'/female', exist_ok=True)

for idx, i in enumerate(sorted(glob.glob(os.path.join(path,'**','*.jpg')))):
    #import pdb;pdb.set_trace()
    if i[-3:] != 'jpg':
        continue

    else:
        img = resize(io.imread(i),(256,256))*255
        landmark = landmarkModel.get(img)
        #upper_folder = i.split('/')[-2]
        #os.makedirs(os.path.join(save_path,upper_folder), exist_ok=True)
        #img_name = i.split('/')[-1]
        #upper_folder = i.splite('/')[-2]
        #os.makedirs()
        np.save(i.replace('val','val_ldmks').replace('.jpg','.npy'),landmark)
        #break
    if idx % 100 == 0:
        print(idx)
        print(i)

#_,_,ldmks_wp_right = mtcnn.detect((wp_right.type(torch.float64).permute(0,2,3,1)+1)/2*255,landmarks=True)