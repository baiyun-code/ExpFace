class cfg:
    # Margin Base Softmax
    margin_list = (1.0, 0.0, 0.4)

    # Partial FC
    sample_rate = 1.0
    interclass_filtering_threshold = 0

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
    seed = 1024  #3407

    # For SGD 
    optimizer = "sgd"
    lr = 0.02
    momentum = 0.9
    weight_decay = 5e-4
    
    # dataload numworkers
    num_workers = 2
    batch_size = 256
    embedding_size = 512

    image_size = (112, 112)
    network = "r50"
    device = "cuda"
    resume = True
    save_all_states = True
    dynamic_margin=False
    output = "../Output/WebFace4M_cosface_0.35_20"
    rec = "../Data/WF4M.pickle"
    val = "../Data/test"
    num_classes = 205990
    num_image = 4235242
    num_epoch = 20
    steps_per_epoch = num_image // batch_size
    total_step = steps_per_epoch * num_epoch
    warmup_epoch = 1
    # ["lfw", "cplfw", "calfw", "cfp_ff", "cfp_fp", "agedb_30", "vgg2_fp"]
    val_targets = ["lfw", "cplfw", "calfw", "cfp_fp", "agedb_30", "vgg2_fp"]