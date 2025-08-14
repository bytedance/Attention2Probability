#This script is based on https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/scripts/train_example.py
import importlib
import json
import argparse

import torch
# from trainer.trainer import SATETainer
# from models.model import *
import os
from retriever.model import *
from retriever.trainer.trainer import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue to train from checkpoints')
    parser.add_argument('--from_ckpt', type=str)
    parser.add_argument('--reset_optim', action='store_true', help='学习率从指定值开始衰减')
    parser.add_argument('--dev_num_max', type=int, help='最大数量')
    parser.add_argument('--dev_metric', type=str, help='评估指标', default='loss')
    parser.add_argument('--maximize_best', action='store_true', help='评估指标越大越好')
    parser.add_argument('--tb_run_tag', type=str, default='run0')
    parser.add_argument('--trainer', type=str, default='trainer')
    parser.add_argument('--trainer_class', type=str, default='RetrieverTainer')
    args = parser.parse_args()
    print(args)
    with open(args.config) as f:
        cfg = json.load(f)
    
    model = Retriever(cfg)
    if cfg.get('pretrained_model') != None:
        print(f"load pretrained model from {cfg.get('pretrained_model')}")
        ckpt = torch.load(cfg.get('pretrained_model'))
        ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=True)
    model.to(torch.bfloat16)
    trainer = RetrieverTainer(model=model, cfg=cfg)

    if args.continue_train:
        trainer.continue_train(cfg=args)
    else:
        trainer.train()