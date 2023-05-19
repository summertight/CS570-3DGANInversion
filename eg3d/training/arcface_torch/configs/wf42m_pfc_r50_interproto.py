from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.2
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.4 # 0.1
config.verbose = 10000
config.dali = False

config.rec = "/home/nas1_userB/dataset/WebFace42M/img_folder"
config.num_classes = 2059906
config.num_image = 42474557
config.num_epoch = 10
config.warmup_epoch = 2
config.val_targets = ["fgnet30_child", "agedb30_child"]

config.save_all_states = True

# TODO: fill in both values
config.head_only = False
config.inter_prototype_file = '/home/nas4_user/jungsoolee/Face_dataset/webface_blur_under_0.04K_20.pickle'
config.child_filter = 10