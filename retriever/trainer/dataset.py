# Copyright (c) 2024 zhangxin
# Copyright (c) 2025 ByteDance Ltd.
# SPDX-License-Identifier: Apache 2.0 license
#
# This file is based on https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/speechtokenizer/trainer/dataset.py
# This file has been modified by ByteDance Ltd. on 27.08.2025
#
# Original file was released under Apache 2.0 license, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import numpy
import torchaudio
import json
try:
    from typing import List,Dict,Unpack
except:
    from typing_extensions import List,Dict,Unpack
import librosa
from transformers.processing_utils import ProcessingKwargs
import random
import torch.multiprocessing as mp
import os
import tarfile
import soundfile as sf
import io

def get_dataloader(ds, speechdatacollator, **kwargs):
    return DataLoader(ds, collate_fn=speechdatacollator, **kwargs)

def read_from_json(file):
    file_reader = open(file,'r')
    json_datas = json.load(file_reader)
    return json_datas


class Qwen2AudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {"sampling_rate": 16000},
    }

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 sample_rate,
                 audio_lang,
                 tgt_lang,
                 pharse_type):
        super().__init__()
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.id_key = "id"
        self.wav_path = "wav_path"
        self.audio_lang = audio_lang
        self.tgt_lang = tgt_lang
        self.pharse_type = pharse_type
        
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, index):
        file = self.file_list[index]
        id_file = file[self.id_key]
        src_text = file[self.audio_lang]
        if isinstance(file['hints'],dict):
            src_terms = [term for term in file["hints"][self.audio_lang]]
            if self.pharse_type and 'tar_path' not in file:
                src_terms = self.process_term(src_terms,src_text)
        else:
            src_terms = [term[self.audio_lang] for term in file["hints"]]
        if 'tar_path' in file:
            tar_path = file['tar_path']
            tar = tarfile.open(tar_path, 'r')
            audio_file = tar.extractfile(file['tar_info'])
            audio_bytes = audio_file.read()
            audio_file = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_file)
            audio = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        else:
            audio_file = file[self.wav_path]
            audio, sr = librosa.load(audio_file,sr=self.sample_rate)

        return {"audio":audio, "src_text":src_text, 'id': id_file, 'src_terms': src_terms}
    
    def process_term(self, src_terms,src_text):
        if len(src_terms) == 1:
            return src_terms
        random.shuffle(src_terms)
        
        nums_terms = len(src_terms)
        text_list = src_text.split(' ')

        for i in range(nums_terms):
            position = text_list.index(src_terms[i])
            length = random.randint(2,4)
            if position + length > len(text_list):
                length = len(text_list) - position
            src_terms[i] = ' '.join(text_list[position:position+length])
        
        return src_terms



class SpeechDataCollator:
    def __init__(self, processor, max_audio_len, max_text_len, max_term_banks, device=None, **kwargs: Unpack[Qwen2AudioProcessorKwargs],):
        self.processor = processor
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.output_kwargs = self.processor._merge_kwargs(
            Qwen2AudioProcessorKwargs,
            tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs,
            **kwargs,
        )
        self.max_term_banks = max_term_banks
        self.device = device

    
    def process_text(self, src_terms) ->dict:
        # import pdb;pdb.set_trace()
        list_src_terms = []
        for src_term in src_terms:
            for term in src_term:
                list_src_terms.append(term)
        combined = list(set(list_src_terms))
        random.shuffle(combined)
        src_term_bank = combined[:self.max_term_banks]
        term_label = torch.zeros((len(src_terms),len(src_term_bank)))

        for i in range(len(src_terms)):
            for term in src_terms[i]:
                if not term in src_term_bank:
                    continue
                index = src_term_bank.index(term)
                term_label[i,index] = 0
                term_label[i,index] = 1
        
        self.output_kwargs["text_kwargs"]['padding_side']='right'
        inputs = self.processor.tokenizer(src_term_bank, **self.output_kwargs["text_kwargs"], add_special_tokens=False, return_tensors="pt")
        attention_mask = inputs['attention_mask']
        input_ids = inputs['input_ids']
        term_label = term_label.long()
        return input_ids, attention_mask, term_label

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_list = [x["audio"] for x in samples]
        src_terms = [x["src_terms"] for x in samples]
        audio_inputs = self.processor.feature_extractor(audio_list, sampling_rate=16000, padding="max_length", return_attention_mask=True)
        input_ids, attention_mask, term_label = self.process_text(src_terms)
        result={
            "audio_list":audio_inputs,
            "input_ids": input_ids,
            "time_word_ids_mask": attention_mask,
            "label": term_label
        }
        
        return result