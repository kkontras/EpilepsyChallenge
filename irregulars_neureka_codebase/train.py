
from Dataloader.tuhseizit2_dataloader import TUHSeizIT2_Dataloader
from tqdm import tqdm
from easydict import EasyDict
import einops
from pytorch_models.neureka_models import *
import torch
from utils.bce_labelsmooth import BinaryCrossEntropyWithLabelSmoothingAndWeights
from collections import defaultdict
import os
from colorama import Fore
from utils.deterministic_pytorch import deterministic
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
import sys
import pathlib
sys.path.insert(0, './')
from utils.SzcoreEvaluation import MetricsStore
import os
from os import path
from utils.postprocessing import post_processing, mask2eventList



def save_model(model_save, is_best_model, unwrapped_model, optimizer, scheduler, logs, config, dataloaders, post_test_results=None, verbose=True):
    save_dict = {}
    savior = {}
    file_name = config.model.save_dir
    if "save_base_dir" in config.model:
        file_name = os.path.join(config.model.save_base_dir, file_name)
    savior["optimizer_state_dict"] = optimizer.state_dict()
    savior["scheduler_state_dict"] = scheduler.state_dict()
    savior["logs"] = logs
    savior["configs"] = config
    if hasattr(dataloaders.train_loader, "generator"):
        savior["training_dataloder_generator_state"] = dataloaders.train_loader.generator.get_state()

    if not model_save:
        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
            if "best_model_state_dict" in prev_checkpoint:
                savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
            if "model_state_dict" in prev_checkpoint:
                savior["model_state_dict"] = prev_checkpoint["model_state_dict"]
    else:
        savior["model_state_dict"] = unwrapped_model.state_dict()

    if is_best_model:
        savior["best_model_state_dict"] = unwrapped_model.state_dict()
        if verbose:
            print(Fore.WHITE + "New best model found!")
    else:
        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
            if "best_model_state_dict" in prev_checkpoint:
                savior["best_model_state_dict"] = prev_checkpoint["best_model_state_dict"]
        else:
            savior["best_model_state_dict"] = unwrapped_model.state_dict()
    if post_test_results:
        savior["post_test_results"] = post_test_results
    if hasattr(dataloaders, "metrics"):
        savior["metrics"] = dataloaders.metrics

    save_dict.update(savior)

    try:
        # accelerator.save(save_dict, file_name)
        torch.save(save_dict, file_name)
        if verbose:
            print(Fore.WHITE + "Models has saved successfully in {}".format(file_name))
    except:
        raise Exception("Problem in model saving")

def load_dir(config, model, optimizer, scheduler, dataloaders):

    file_name = config.model.save_dir
    if "save_base_dir" in config.model:
        file_name = os.path.join(config.model.save_base_dir, file_name)

    checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logs = checkpoint["logs"]
    config = checkpoint["configs"]
    # if hasattr(dataloaders.train_loader, "generator"):
    #     dataloaders.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
    if "metrics" in checkpoint:
        dataloaders.metrics = checkpoint["metrics"]
    print(Fore.WHITE + "Model has loaded successfully from {}".format(file_name))
    return model, optimizer, scheduler, logs, config, dataloaders

def load_encoder(enc_args, config):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = globals()[enc_args[num_enc]["model_class"]]
            args = enc_args[num_enc]["args"]
            if "encoders" in enc_args[num_enc]:
                enc_enc = load_encoder(enc_args[num_enc]["encoders"], config)
                enc = enc_class(encs = enc_enc, args = args)
            else:
                enc = enc_class(args = args, encs=[])
            pretrained_encoder_args =  enc_args[num_enc].get("pretrainedEncoder", {"use":False})
            if pretrained_encoder_args["use"]:
                config.model.save_base_dir = config.model.get("save_base_dir", "")
                file_path = path.join(config.model.save_base_dir, pretrained_encoder_args.get("dir",""))
                print("Loading encoder from {}".format(file_path))
                if "save_base_dir" in config.model:
                    file_path = os.path.join(config.model.save_base_dir, file_path)
                checkpoint = torch.load(file_path)
                if "encoder_state_dict" in checkpoint:
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

                elif "best_model_state_dict" in checkpoint:
                    missing_keys, unexpected_keys =  enc.load_state_dict(checkpoint["best_model_state_dict"], strict=False)
                    if missing_keys:
                        print(f"Missing keys in state_dict: {missing_keys}")
                    if unexpected_keys:
                        print(f"Unexpected keys in state_dict: {unexpected_keys}")

            encs.append(enc)
        return encs

def load_model(config):
    enc = load_encoder(enc_args=config.model.get("encoders", []), config=config)
    model_class = globals()[config.model.model_class]
    model = model_class(encs=enc, args=config.model.args)
    model.cuda()
    return model

def load_optimizer_and_scheduler(config, model):

    if config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=(config.optimizer.beta1, config.optimizer.beta2))
    else:
        raise Exception("Optimizer type not supported")

    if config.scheduler.type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.T_0, T_mult=config.scheduler.T_mult)
    else:
        raise Exception("Scheduler type not supported")

    return optimizer, scheduler

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def validate(model, dataloaders, logs, epoch, loss, config, set_name="val"):


    # Validation loop
    model.eval()

    total_preds = defaultdict(list)
    pbar = tqdm(enumerate(dataloaders), total=len(dataloaders))
    for current_step, batch in pbar:
        data = batch['data']['raw'].cuda().float()
        label = batch['label'].cuda()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)

        preds = model(data)

        losses = {}
        for key_pred, pred in preds.items():
            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze()

            total_preds["{}_label".format(key_pred)].append(this_label.detach().cpu().numpy())
            total_preds[key_pred].append(pred.detach().cpu().numpy())

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss

        total_preds["demographics"].append(batch["demographics"])

        total_loss = losses[0] + 0.2 * (torch.tensor([v for k, v in losses.items() if k != 0]).sum())

        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}

        if epoch in logs:
            logs[epoch][set_name]["losses"].append(total_loss.item())
            if set_name == "test":
                logs["best"]["test_loss"] = logs[epoch][set_name]["losses"]

        aggr_loss = {key: torch.tensor(val).mean().item() for key, val in losses.items()}
        message = "{0:} Epoch {1:} step {2:} with ".format(set_name, epoch, current_step)
        for i, v in aggr_loss.items(): message += "{} : {:.6f} ".format(i, v)
        pbar.set_description(message)
        pbar.refresh()
        print(current_step)
        if current_step > 100:
            break


    # merge predictions from same patients in the correct order for key=0 not in one line
    aggregated_patient_preds = {}
    for pred, label, demo in zip(total_preds[0], total_preds["0_label"], total_preds["demographics"]):
        for i in range(len(demo["patient"])):
            name = demo["patient"][i] + "_" + demo["session"][i] + "_" + demo["recording"][i]
            len_from = demo["len_from"][i]
            len_to = demo["len_to"][i]
            if name not in aggregated_patient_preds:
                aggregated_patient_preds[name] = {"preds": {}, "label": {}}
            aggregated_patient_preds[name]["preds"]["{}_{}".format(len_from, len_to)] = pred[i]
            aggregated_patient_preds[name]["label"]["{}_{}".format(len_from, len_to)] = label[i]

    metricsStoreTest = MetricsStore(config)
    #sort the pred len_from, len_to and merge them into one array
    for key, val in aggregated_patient_preds.items():
        sorted_keys = sorted(val["preds"].keys(), key=lambda x: int(x.split("_")[0]))
        val["preds"] = np.concatenate([val["preds"][k] for k in sorted_keys], axis=0)
        val["label"] = np.concatenate([val["label"][k] for k in sorted_keys], axis=0)
        classification_threshold = config.model.args.get("cls_threshold",0.5)

        event_preds = post_processing(val["preds"], 200, 0.5, 10)
        event_labels = mask2eventList(val["label"], 200)
        print(key)
        print(event_preds)
        print(event_labels)
        print()

        #check if preds are nan
        metricsStoreTest.evaluate_multiple_predictions(val["label"], (val["preds"] > classification_threshold), key.split("_")[0])
        # metricsStoreTest.evaluate_multiple_predictions_windows(loaded_events, events, key.split("_")[0])

    results_per_patient = metricsStoreTest.store_scores()
    metrics = metricsStoreTest.store_metrics()

    for total_size in [200]: #fs=200 so 5, 2, 1, 0.5 seconds
        this_label = np.concatenate(total_preds["{}_label".format(0)])

        #TODO: Remove this necessity to transform to torch Tensor and back in numpy
        this_label = torch.nn.functional.interpolate(torch.from_numpy(this_label).unsqueeze(dim=1), size=(total_size), mode='nearest').flatten().numpy()

        unique, counts = np.unique(this_label, return_counts=True)
        if len(unique) == 1:
            print("Label percentage 0: {:.2f}% 1: {:.2f}%".format(1, 0))
        else:
            print("Label percentage 0: {:.2f}% 1: {:.2f}%".format(counts[0]/len(this_label), counts[1]/len(this_label)))

    message = "{0:} Epoch {1:} step {2:} with \n".format(set_name, epoch, current_step)
    for i, v in metrics.items():
        message += "Type of results: {0:} \n".format(i)
        for key, val in v.items():
            message += "{} : {:.3f} ".format(key, val)
        message += "\n"
    print(message)
    logs[epoch][set_name]["metrics"] = metrics

    return logs

def is_best(logs, epoch, config):

    this_epoch_val_f1 = logs[epoch]["val"]["metrics"]["event_results"]["f1"]
    this_epoch_val_loss = torch.tensor(logs[epoch]["val"]["losses"]).mean()
    is_best_model = False
    if config.model.args.get("validate_with", "f1") == "f1":
        if this_epoch_val_f1 > logs["best"]["f1"]:
            logs["best"]["f1"] = this_epoch_val_f1
            logs["best"]["loss"] = this_epoch_val_loss
            logs["best"]["epoch"] = epoch
            is_best_model = True
    elif config.model.args.get("validate_with", "f1") == "loss":
        if this_epoch_val_loss < logs["best"]["loss"]:
            logs["best"]["f1"] = this_epoch_val_f1
            logs["best"]["loss"] = this_epoch_val_loss
            logs["best"]["epoch"] = epoch
            is_best_model = True
    return is_best_model

def train_loop(epoch, model, optimizer, scheduler, loss, dataloader, logs, config):

    # Training loop
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=None, )
    for current_step, batch in pbar:
        optimizer.zero_grad()
        data = batch['data']['raw'].cuda().float()
        label = batch['label'].cuda()
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1)

        preds = model(data)
        losses = {}

        for key_pred, pred in preds.items():
            this_label = einops.rearrange(label, "b t -> b 1 t")
            this_label = torch.nn.functional.interpolate(this_label, size=(pred.shape[1]), mode='nearest').squeeze()

            this_pred_loss = loss(pred, this_label)
            losses[key_pred] = this_pred_loss

        total_loss = losses[0] + 0.2 * (torch.tensor([v for k, v in losses.items() if k != 0]).sum())

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses["total"] = total_loss
        losses = {key: loss.item() for key, loss in losses.items()}
        logs[epoch]["train"]["losses"].append(total_loss.item())


        aggr_loss = {key: torch.tensor(val).mean().item() for key, val in losses.items()}
        message = "Train Epoch {0:d} step {1:d} with ".format(epoch, current_step)
        for i, v in aggr_loss.items(): message += "{} : {:.6f} ".format(i, v)
        pbar.set_description(message)
        pbar.refresh()

        # threshold = 0.5
        # binary_preds = (preds[0] > threshold)
        # binary_preds = torch.nn.functional.one_hot(binary_preds.long().cuda().flatten(), num_classes=2).float()
        #
        # if current_step == 10:
        #     break

    return logs

def load_best_model(model, config):
    file_name = path.join(config.model.save_base_dir, config.model.save_dir)
    if os.path.exists(file_name):
        print(Fore.WHITE + "Loading best model from {}".format(file_name))
        checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["best_model_state_dict"])
    return model


def main_train(config):
    deterministic(config.training_params.seed)
    dataloader_class = globals()[config.dataset.dataloader_class]
    dataloaders = dataloader_class(config)

    model = load_model(config)
    optimizer, scheduler = load_optimizer_and_scheduler(config, model)

    loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()

    file_name = path.join(config.model.save_base_dir, config.model.save_dir)
    if file_name and config.model.get("load_ongoing", True):
        model, optimizer, scheduler, logs, config, dataloaders = load_dir(config, model, optimizer, scheduler, dataloaders)
    else:
        logs = {"best": {"loss": 1e20, "f1": 0.0}, "epoch":0}

    try:

        # logs[-1] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
        # _ = validate(model, dataloaders.valid_loader, logs, -1, loss, config, "val")

        for logs["epoch"] in range(logs["epoch"], config.early_stopping.max_epoch):
            if logs["epoch"] not in logs:
                logs[logs["epoch"]] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

            logs = train_loop(logs["epoch"], model, optimizer, scheduler, loss, dataloaders.train_loader, logs, config)

            logs = validate(model, dataloaders.valid_loader, logs, logs["epoch"], loss, config, "val")

            save_model(True,
                       is_best(logs, logs["epoch"], config),
                       model, optimizer, scheduler, logs, config, dataloaders)
    #catch a control c and save the model
    except KeyboardInterrupt:
        save_model(True, False, model, optimizer, scheduler, logs, config, dataloaders)
        raise KeyboardInterrupt

    model = load_best_model(model, config)
    logs = validate(model, dataloaders.valid_loader, logs, "post_training", loss, config, "val")
    # logs = validate(model, dataloaders.test_loader, logs, logs["epoch"], loss, config, "test")
    save_model(False,
               False,
               model, optimizer, scheduler, logs, config, dataloaders)

def main_validate(config):
    deterministic(config.training_params.seed)
    dataloader_class = globals()[config.dataset.dataloader_class]
    config.training_params.batch_size = 16
    config.training_params.test_batch_size = 16
    dataloaders = dataloader_class(config)

    model = load_model(config)
    optimizer, scheduler = load_optimizer_and_scheduler(config, model)

    loss = BinaryCrossEntropyWithLabelSmoothingAndWeights()

    file_name = path.join(config.model.save_base_dir, config.model.save_dir)
    if os.path.exists(file_name) and config.model.get("load_ongoing", True):
        model, optimizer, scheduler, logs, config, dataloaders = load_dir(config, model, optimizer, scheduler, dataloaders)
    else:
        logs = {"best": {"loss": 1e20}, "epoch":0}

    model = load_best_model(model, config)
    logs["post_training"] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    logs = validate(model, dataloaders.valid_loader, logs, "post_training", loss, config, "val")
    # logs = validate(model, dataloaders.test_loader, logs, "post_training", loss, config, "test")
    save_model(False,
               False,
               model, optimizer, scheduler, logs, config, dataloaders)


def check_results(config):

    file_name = config.model.save_dir
    if "save_base_dir" in config.model:
        file_name = os.path.join(config.model.save_base_dir, file_name)
    print("Checking file {}".format(file_name))
    try:
        checkpoint = torch.load(file_name, map_location="cpu", weights_only=False)
    except:
        print("Couldnt load file {}".format(file_name))
        return
    if "best" in checkpoint["logs"]:
        # print("Best so far")
        # print(checkpoint["logs"]["best"])
        if "epoch" in checkpoint["logs"]["best"]:
            print("Current Epoch {} Best Epoch {}".format(checkpoint["logs"]["epoch"], checkpoint["logs"]["best"]["epoch"]), end=" ")
            print("f1 {:.4f}".format(checkpoint["logs"][checkpoint["logs"]["best"]["epoch"]]["val"]["metrics"]["event_results"]["f1"]), end=" ")
            print("sens {:.4f}".format(checkpoint["logs"][checkpoint["logs"]["best"]["epoch"]]["val"]["metrics"]["event_results"]["sensitivity"]))
            # print("f1 {:.4f}".format(checkpoint["logs"][checkpoint["logs"]["best"]["epoch"]]["val"]["metrics"]["sample_results"]["f1"]), end=" ")
            # print("sens {:.4f}".format(checkpoint["logs"][checkpoint["logs"]["best"]["epoch"]]["val"]["metrics"]["sample_results"]["sensitivity"]))
        else:
            #find the valudation epoch with the same loss
            for epoch, logs in checkpoint["logs"].items():
                #if epoch is int
                if isinstance(epoch, int):
                    print("Epoch {}".format(epoch))
                    print(logs["val"]["metrics"]["event_results"]["f1"])
                    print(logs["val"]["metrics"]["event_results"]["sensitivity"])
    if "post_training" in checkpoint["logs"]:
        print("Post training")
        print(checkpoint["logs"]["post_training"])
