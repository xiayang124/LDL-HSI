import os
from engine import train
from engine.config import load_config
import utils as utils
import logger.log as log
import argparse
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--dataset", type=str, default="Honghu", help="Dataset name", choices=["PaviaU", "Houston", "Honghu"])
    parser.add_argument("--change_text", type=bool, default=False)
    args = parser.parse_args()
    
    dataset = args.dataset
    
    # Get Setting
    setting_loc = f"./config/{dataset}.json"

    cfg = load_config(setting_loc)
    cfg.BaseSetting.set("change_text", args.change_text)

    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_save_loc = "./result/" + dataset + "_" + datetime_now + ".txt"
    utils.ensure_dir("./result/")
    recoding = log.Logger(log_path=log_save_loc)

    train.choose_trainer(cfg, recoding).train()
