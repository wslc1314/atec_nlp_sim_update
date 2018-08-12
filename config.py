class Config:

    global_train_file="data/atec/training.csv"
    global_t2s_dict="data/atec/t2s_dict.json"
    local_train_file="data/atec/10/train0.csv"
    local_valid_file="data/atec/10/valid0.csv"
    min_count_wc=[2,2]
    global_dict="data/atec/training-"+"-".join([str(i) for i in min_count_wc])+".json"
    local_dict="data/atec/10/train0-"+"-".join([str(i) for i in min_count_wc])+".json"

    focal=None
    alpha=None
    gamma=2

    threshold = 0.5

    dropout = None
    fine_tune = None
    init_learning_rate = None

    with_validation = None
    train_file = None
    valid_file = None

    load_path = None

    max_to_keep = None

    num_epochs = None
    steps_every_epoch = None
    batch_size = None
    save_epochs = None
    early_stopping = None
