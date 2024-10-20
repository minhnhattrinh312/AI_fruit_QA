import glob
import torch
from torch.utils.data import Dataset, DataLoader
# make argument parser
import argparse
import pandas as pd
from classification import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch._dynamo
torch._dynamo.config.suppress_errors = True

def normalize_01(data):
    return (data - data.min()) / (data.max() - data.min())


class Get_samples(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # the data is numpy array
        # expend the dimension the data from (n,) to (1, n)
        data = self.data[idx][None]
        return torch.tensor(data, dtype=torch.float32)

# make task argument
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="brix", help="task to predict (brix or firm)")

task = parser.parse_args().task
weight_brix = sorted(glob.glob(f"weights_{task}/*/*"))


model = BasicModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier(model, cfg.OPT.LEARNING_RATE, cfg.OPT.FACTOR_LR, cfg.OPT.PATIENCE_LR)
x_test_df = pd.read_csv("challenge_data/x_2d_snv_test.csv", index_col=0)
x_test = np.array(x_test_df)
x_test = normalize_01(x_test)

test_dataset = Get_samples(x_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    num_workers=cfg.TRAIN.NUM_WORKERS,
    prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
)
classifier = classifier.to(device)

if __name__ == "__main__":
    sum_predictions = 0
    with torch.inference_mode():
        for weight in weight_brix:
            test_predictions = []
            classifier = Classifier.load_from_checkpoint(
                checkpoint_path=weight,
                model=model,
                learning_rate=cfg.OPT.LEARNING_RATE,
                factor_lr=cfg.OPT.FACTOR_LR,
                patience_lr=cfg.OPT.PATIENCE_LR,
            )
            classifier.eval()
            for data in tqdm(test_loader, desc="Prediction Loop"):
                # print(data)
                pred = classifier(data.to(device)).cpu().numpy()[:, 0]
                test_predictions.append(pred)
            test_predictions = np.concatenate(test_predictions)
            sum_predictions += test_predictions
        sum_predictions /= len(weight_brix)

test_pred = pd.DataFrame(sum_predictions, index=x_test_df.index, columns=[f"{task}"])
test_pred.to_csv(f"{task}_test_pred_Nhat.csv")
