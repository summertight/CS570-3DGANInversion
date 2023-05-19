import imageio as io
import torch, os
import numpy as np
from skimage.transform import resize
#from facenet_pytorch import MTCNN, InceptionResnetV1
import sys
sys.path.append('./eg3d.training.facenet_pytorch_folder')
from training.facenet_pytorch_folder.models.mtcnn import MTCNN

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=(256), margin=0)
path = './inthewild_datav2'
path = './sample_refined'

os.makedirs(path+'_kps',exist_ok=True)
for i in os.listdir(path):
    if i[-3:] != 'png':
        continue

    else:
        img_temp  = torch.tensor(resize(io.imread(os.path.join(path,i)),(256,256)),dtype=torch.float32).unsqueeze(0)
        _, ldmk_tensor = mtcnn.detect_tensor(img_temp*255,landmarks=True)
        _,_,ldmk=mtcnn.detect(img_temp*255,landmarks=True)
        breakpoint()
        from facenet_pytorch import MTCNN, InceptionResnetV1
        mtcnn = MTCNN(image_size=(256), margin=0)
        _,_,ldmk_org=mtcnn.detect(img_temp*255,landmarks=True)
        #np.save(path+'_kps/'+i,ldmk.squeeze(0))
        #break

#_,_,ldmks_wp_right = mtcnn.detect((wp_right.type(torch.float64).permute(0,2,3,1)+1)/2*255,landmarks=True)