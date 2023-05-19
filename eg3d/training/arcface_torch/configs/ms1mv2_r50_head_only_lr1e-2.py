from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.01
config.verbose = 2000
config.dali = False

# New for freezing backbone
config.head_only = True
config.save_all_states = True
config.pretrained_path = "/home/nas1_userC/jooyeolyun/repos/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc_r50/model.pt"
config.lr_scheduler = 'fixed'

#config.rec = "/home/nas4_user/jungsoolee/face_dataset/ms1m-refined-112/ms1m"
config.rec = "/home/jungsoolee/ms1m-refined-112/ms1m"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 10
config.warmup_epoch = 0
config.val_targets = ["agedb30_child", "fgnet30_child"]
