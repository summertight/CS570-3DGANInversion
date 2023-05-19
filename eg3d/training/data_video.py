from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np
class CustomDataset(Dataset):
    def __init__(self,root):
    # 생성자, 데이터를 전처리 하는 부분
        self.root = root   
        self.img_dir_list = os.path.listdir(root)
       #ldmk_dir_list = os.path.listdir(path.replace('crop','crop_kps').replace)

    def __len__(self):
    # 데이터셋의 총 길이를 반환하는 부분   
        return len(self.img_dir_list)

    def __getitem__(self,idx):
    # idx(인덱스)에 해당하는 입출력 데이터를 반환한다.
        img_path = os.path.join(self.root,self.img_dir_list[idx])
        img = np.array(Image.open(img_path)).transpose(2,0,1)
        msk = np.array(Image.open(img_path.replace('crop','crop_msks')))[...,None].transpose(2,0,1)
        ldmk = np.load(img_path.replace('crop','crop_kps'))