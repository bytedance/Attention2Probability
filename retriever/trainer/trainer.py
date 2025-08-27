# Copyright (c) 2024 zhangxin
# Copyright (c) 2025 ByteDance Ltd.
# SPDX-License-Identifier: Apache 2.0 license
#
# This file is based on https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/speechtokenizer/trainer/trainer.py
# This file has been modified by ByteDance Ltd. on 27.08.2025
#
# Original file was released under Apache 2.0 license, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.
import sys
import os


from beartype import beartype
from retriever.model import Retriever

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration
import json
import time
from tqdm import tqdm
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from retriever.trainer.dataset import *
from retriever.trainer.optimizer import *
from torch.nn.utils import clip_grad_norm_
from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class RetrieverTainer(nn.Module):
    @beartype
    def __init__(
        self,
        model: Retriever,
        cfg,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        torch.manual_seed(cfg.get('seed'))
        split_batches = cfg.get("split_batches", False)
        self.stdout_steps = cfg.get('stdout_steps')
        self.save_model_steps = cfg.get('save_model_steps')
        results_folder = cfg.get('results_folder')
        self.results_folder = Path(results_folder)
        self.num_ckpt_keep = cfg.get("num_ckpt_keep")
        self.epochs = cfg.get("epochs")
        self.num_warmup_steps = cfg.get("num_warmup_steps")
        self.batch_size = cfg.get("batch_size")
        self.showpiece_num = cfg.get('showpiece_num', 8)
        project_name = 'Retriever'
        if not self.results_folder.exists():
            self.results_folder.mkdir(parents = True, exist_ok = True)
        
        with open(f'{str(self.results_folder)}/config.json', 'w+') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
            
    
        # tracker = AudioTensorBoardTracker(run_name=project_name, logging_dir=results_folder)
        dataloader_config = DataLoaderConfiguration(split_batches=split_batches) 
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision="bf16",
            # log_with=tracker,
            **accelerate_kwargs
        )

        self.register_buffer('steps', torch.Tensor([0]))

        

        # data
        # self.max_grad_norm = max_grad_norm
        self.sample_rate=cfg.get("sample_rate")
        batch_size = cfg.get("batch_size")
        self.batch_size = batch_size
        train_files = cfg.get("train_files")
        valid_files = cfg.get("valid_files")
        train_file_list = read_from_json(train_files)
        valid_file_list = read_from_json(valid_files)
        
        audio_lang = cfg.get("audio_lang")
        tgt_lang = cfg.get("tgt_lang")
        self.ds = audioDataset(file_list=train_file_list,
                                sample_rate=self.sample_rate,
                                audio_lang=audio_lang,
                                tgt_lang=tgt_lang,
                                pharse_type=True)
        self.valid_ds = audioDataset(file_list=valid_file_list,
                                    sample_rate=self.sample_rate,
                                    audio_lang=audio_lang,
                                    tgt_lang=tgt_lang,
                                    pharse_type=True)
        if self.is_main:
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            
        assert len(self.ds) >= self.batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= self.batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader
        drop_last = cfg.get("drop_last", True)
        num_workers = cfg.get("num_workers")
        self.processor = AutoProcessor.from_pretrained(cfg.get("qwen_path"))
        self.audio_encoder = torch.load(cfg.get('audio_encoder_path'))
        self.audio_projector = torch.load(cfg.get('audio_projector_path'))
        self.model_embedding = torch.load(cfg.get('model_embedding_path'))
        self.max_term_banks = cfg.get('max_term_banks')
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_projector.parameters():
            param.requires_grad = False
        for param in self.model_embedding.parameters():
            param.requires_grad = False
        data_collator = SpeechDataCollator(processor=self.processor, max_audio_len=300000, max_text_len=2048, max_term_banks=self.max_term_banks)
        
        # num_workers = 0
        self.dl = get_dataloader(self.ds, data_collator, batch_size = self.batch_size, shuffle = True, drop_last = drop_last, num_workers=num_workers)
        self.valid_dl = get_dataloader(self.valid_ds, data_collator, batch_size = self.batch_size, shuffle = False, drop_last = False, num_workers=num_workers)
        
        # lr
        self.lr = cfg.get("learning_rate")
        self.initial_lr = cfg.get("intial_learning_rate")

        # model
        self.model=model
        # optimizer
        self.optim = get_optimizer(
            self.model.parameters(),
            lr = self.lr,
            wd = cfg.get("wd"),
            betas = cfg.get("betas")
        )
        

        num_train_steps = self.epochs * self.ds.__len__() // batch_size
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)


        (
            self.model,
            self.audio_encoder,
            self.audio_projector,
            self.model_embedding,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.model,
            self.audio_encoder,
            self.audio_projector,
            self.model_embedding,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )
        
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": self.num_warmup_steps, "learning_rate": self.lr, "initial_learning_rate": self.initial_lr, "epochs": self.epochs}
        self.accelerator.init_trackers("Retriever", config=hps)
        self.best_dev_loss = float('inf')
        self.plot_gt_once = False


    def load(self, path = None, restore_optimizer = True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        model = self.accelerator.unwrap_model(self.model)
        pkg = torch.load(path, map_location='cpu')
        model.load_state_dict(pkg['model'])

        if restore_optimizer:
            self.optim.load_state_dict(pkg['optim'])
            self.scheduler.load_state_dict(pkg['scheduler'])
            if 'best_dev_loss' in pkg.keys():
                self.best_dev_loss = pkg['best_dev_loss']
                if self.is_main:
                    self.print(f'The best dev loss before is {self.best_dev_loss}')

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([1000 + 1], device=self.device)

    def save(self, path, best_dev_loss):
        if best_dev_loss < self.best_dev_loss:
            self.best_dev_loss = best_dev_loss
            torch.save(self.accelerator.get_state_dict(self.model), f'{self.results_folder}/Retriever_best_dev.pt')
        ckpts = sorted(Path(path).parent.glob(f'Retriever_*'),reverse=True)
        if len(ckpts) > self.num_ckpt_keep:
            [os.remove(c) for c in ckpts[-self.num_ckpt_keep:-self.num_ckpt_keep+1]] # remove the last one but not best_dev
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict(),
            best_dev_loss = self.best_dev_loss
        )
        torch.save(pkg, path)


    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

        
    def process_audio(self, audio_data) -> dict:
        """Load and preprocess a single audio file"""
        feature_attention_mask = audio_data['attention_mask']
        input_features = audio_data['input_features']
        input_features = torch.from_numpy(input_features)
        feature_attention_mask = torch.from_numpy(feature_attention_mask)
        audio_feat_lengths, audio_output_lengths = self.audio_encoder._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_encoder.conv1.weight.dtype, device=self.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        input_features = input_features.to(
            dtype=self.audio_encoder.conv1.weight.dtype, device=self.device
        )
        audio_outputs = self.audio_encoder(input_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_projector(selected_audio_feature)
        audio_attention_mask = torch.ones((audio_features.shape[0],audio_features.shape[1])).to(self.device)

        return audio_features, audio_attention_mask
    
    def process_data(self,batch):
        audio_features, audio_attention_mask = self.process_audio(batch['audio_list'])
        input_embedding = self.model_embedding(batch['input_ids'])
        batch["audio_embeds"] = audio_features
        batch["audio_embeds_mask"] = audio_attention_mask
        batch["time_words_embeds"] = input_embedding
        del batch['audio_list']
        del batch['input_ids']
        return batch
        

    def train(self):
        
        self.model.train()
        step_time_log = {}
        # import pdb;pdb.set_trace()
        steps = int(self.steps.item())
        # import pdb;pdb.set_trace()               
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
        
        for epoch in range(self.epochs):
            if self.is_main:
                print(f'Epoch:{epoch} start...')
                    
            for batch in self.dl:
                tic = time.time()
                batch = self.process_data(batch)
                self.optim.zero_grad() 
                loss = self.model.forward(time_words_embeds=batch['time_words_embeds'], 
                                          time_word_ids_mask=batch['time_word_ids_mask'],
                                          audio_embeds=batch['audio_embeds'], 
                                          audio_embeds_mask=batch['audio_embeds_mask'], 
                                          is_inference=False,
                                          label=batch['label'])

                acc = loss['acc']
                
                loss = loss['backward_loss']
                self.accelerator.backward(loss)

                self.optim.step()
                
                step_time_log = accum_log(step_time_log, {'time_cost': time.time() - tic})
                
                if not (steps % self.stdout_steps):
                    torch.cuda.empty_cache()
                    if self.is_main:
                        self.print(f"Epoch {epoch} -- Step {steps}: Loss: {loss.item():0.3f} acc: {acc:0.3f} lr: {lr:0.7f} Time cost per step: {step_time_log['time_cost'] / self.stdout_steps:0.3f}s")
                        step_time_log = {}
                    
                self.accelerator.wait_for_everyone()
                
                # validate and save model
                if self.is_main and not(steps % self.save_model_steps) and steps != 0:
                    self.print('Validation start ...')
                    # validate
                    total_loss = 0.0
                    total_acc = 0.0
                    num = 0
                    self.model.eval()
                    with torch.inference_mode():
                        for i, batch in tqdm(enumerate(self.valid_dl),disable=True):
                            batch = self.process_data(batch)
                            loss = self.model.forward(time_words_embeds=batch['time_words_embeds'], 
                                          time_word_ids_mask=batch['time_word_ids_mask'],
                                          audio_embeds=batch['audio_embeds'], 
                                          audio_embeds_mask=batch['audio_embeds_mask'], 
                                          is_inference=False,
                                          label=batch['label'])
                            acc = loss['acc']
                            loss = loss['loss']
                            total_acc += acc
                            total_loss += loss.item()                
                            num += 1
                        if not self.plot_gt_once:
                            self.plot_gt_once = True
                        self.print(f'{steps}: dev loss: {total_loss / num:0.3f}; dev acc: {total_acc / num:0.3f}')
                            
                    
                    # save model
                    model_path = str(self.results_folder / f'Retriever_{total_loss / num:0.3f}')
                    self.save(model_path, total_loss / num)                        
                    self.print(f'{steps}: saving model to {str(self.results_folder)}')
                    self.model.train()
                    
                # Update lr    
                self.steps += 1
                steps = int(self.steps.item())               
                if steps < self.num_warmup_steps:
                    lr = self.warmup(steps)
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler.step() 
                    lr = self.scheduler.get_last_lr()[0]    
            
        self.print('training complete')
        
    def continue_train(self):
        self.load(path='/path/save_model')
        self.train()
        