import torch
from torch import nn
#from configs.paths_config import model_paths
from training.criterion_id.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/model_ir_se50.pth')) #XXX Hard coded 
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        #t = 35 - 18
        #b = 223 + 18
        #l = 32 - 18
        #r = 220 + 18

        #x = x[:, :, t:b, l:r] # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)

        return x_feats

    def forward(self, x, y):
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        x_feats = self.extract_feats(x)
        #print(y_feats.shape, x_feats.shape, 'dasd')
        #y_feats = y_feats.detach()
    

        loss = (1 - torch.cosine_similarity(x_feats, y_feats)).mean()

        return loss 

    def forward_contrastive(self, src, tgt, rec):
        src_feats = self.extract_feats(src)  # Otherwise use the feature from there
        tgt_feats = self.extract_feats(tgt)

        rec_feats = self.extract_feats(rec)

        loss = (torch.cosine_similarity(rec_feats, tgt_feats) - torch.cosine_similarity(src_feats, tgt_feats))**2
        #print(y_feats.shape, x_feats.shape, 'dasd')
        #y_feats = y_feats.detach()
    

        

        return loss 