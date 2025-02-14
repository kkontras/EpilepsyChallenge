
from tqdm import tqdm
import einops
from pytorch_models.neureka_models import *
from utils.bce_labelsmooth import BinaryCrossEntropyWithLabelSmoothingAndWeights
from collections import defaultdict
import os
from colorama import Fore
from utils.deterministic_pytorch import deterministic
import numpy as np
import sys
sys.path.insert(0, './')
import os
from os import path
from utils.postprocessing import post_processing, mask2eventList
from train import load_best_model, load_model, save_model, load_optimizer_and_scheduler, load_dir

def validate(model, dataloader, config):


    # Validation loop
    model.eval()

    total_preds = defaultdict(list)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for current_step, batch in pbar:
        data = batch['data']['raw'].cuda().float()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)
        preds = model(data)
        total_preds["idxs"].append(batch['idx'])

        for key_pred, pred in preds.items():
            total_preds[key_pred].append(pred.detach().cpu().numpy())

    
    # merge predictions from same patients in the correct order for key=0 not in one line
    aggregated_preds = []
    for pred, idxs in zip(total_preds[0], total_preds["idxs"]):
        for i in range(len(idxs)):
            aggregated_preds.append((idxs[i], pred[i]))


    #sort the pred len_from, len_to and merge them into one array
    aggregated_preds = sorted(aggregated_preds, key=lambda x: x[0])
    aggregated_preds = [v[1] for v in aggregated_preds]
    aggregated_preds = np.concatenate(aggregated_preds)
    classification_threshold = config.model.args.get("cls_threshold",0.5)

    events = post_processing(aggregated_preds, fs=config.dataset.fs, th=classification_threshold, margin=10)

    return events

def main_validate(config, dataloader):
    deterministic(config.training_params.seed)

    model = load_model(config)
    optimizer, scheduler = load_optimizer_and_scheduler(config, model)

    file_name = path.join(config.model.save_base_dir, config.model.save_dir)
    if os.path.exists(file_name) and config.model.get("load_ongoing", True):
        model, optimizer, scheduler, logs, config, dataloaders = load_dir(config, model, optimizer, scheduler, None)
    else:
        logs = {"best": {"loss": 1e20}, "epoch":0}

    model = load_best_model(model, config)

    logs["post_training"] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    return validate(model, dataloader, config)

    # logs = validate(model, dataloaders.test_loader, logs, "post_training", loss, config, "test")
    # save_model(False,
    #            False,
    #            model, optimizer, scheduler, logs, config, dataloaders)
