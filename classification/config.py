from yacs.config import CfgNode as CN


cfg = CN()
cfg.DATA = CN()
cfg.TRAIN = CN()
cfg.SYS = CN()
cfg.OPT = CN()
cfg.DIRS = CN()
cfg.PREDICT = CN()



cfg.TRAIN.TASK = "brix" # "brix" or "firmness"
# cfg.TRAIN.FOLDS = [1, 2, 3, 4, 5]  # default [1, 2, 3, 4, 5]
cfg.TRAIN.FOLDS = [3]  # default [1, 2, 3, 4, 5]
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.PREFETCH_FACTOR = 2
cfg.TRAIN.WANDB = True
cfg.TRAIN.EPOCHS = 300
cfg.TRAIN.IDX_CHECKPOINT = -1
cfg.TRAIN.LOAD_CHECKPOINT = True
cfg.TRAIN.SAVE_TOP_K = 5

cfg.DIRS.SAVE_DIR = f"./weights_{cfg.TRAIN.TASK}/"

cfg.OPT.LEARNING_RATE = 1e-3
cfg.OPT.FACTOR_LR = 0.5
cfg.OPT.PATIENCE_LR = 10
cfg.OPT.PATIENCE_ES = 50

cfg.SYS.ACCELERATOR = "gpu"
cfg.SYS.DEVICES = [0]
cfg.SYS.MIX_PRECISION = "32"  # "16-mixed"  # 32 or 16-mixed

cfg.PREDICT.FOLD = 1
cfg.PREDICT.ENSEMBLE = True
