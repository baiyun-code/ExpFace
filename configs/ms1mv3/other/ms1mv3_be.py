class cfg:
    # Margin Base Softmax
    margin_list = (1.0, 0.5, 0.0)

    # Partial FC
    sample_rate = 0.5
    interclass_filtering_threshold = 0
    bottle_neck = 32
    fp16 = True

    # For AdamW
    # optimizer = "adamw"
    # lr = 0.001
    # weight_decay = 0.1

    verbose = 16000
    frequent = 10

    # For Large Sacle Dataset, such as WebFace42M
    dali = False 
    dali_aug = False

    # Gradient ACC
    gradient_acc = 1

    # setup seed
    seed = 3407  #3407

    # For SGD 
    optimizer = "sgd"
    lr = 0.02
    momentum = 0.9
    weight_decay = 5e-4
    
    # dataload numworkers
    num_workers = 4
    batch_size = 128
    embedding_size = 512
    s=64.0

    image_size = (112, 112)
    network = "r50"
    resume = False
    save_all_states = True
    device = "cuda"
    output = "Output/ms1mv3_bce3_cosface_64_0.4_30_1" #"Output/ms1mv3" _tanface_0.7_1_grad ms1mv3_arcFace_forward_back_s
    rec = "../Data/ms1mv3.pickle" # "../Data/ms1mv3"
    val = "../Data/test"
    num_classes = 93431
    num_image = 5179510
    num_epoch = 5
    steps_per_epoch = num_image // batch_size
    total_step = steps_per_epoch * num_epoch
    warmup_epoch = 0
    # ["lfw", "cplfw", "calfw", "cfp_ff", "cfp_fp", "agedb_30", "vgg2_fp"]
    val_targets = ["lfw", "cfp_fp", "agedb_30"]