# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration,AutoTokenizer,AutoModelForCausalLM,WhisperProcessor
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperConfig
from retriever.model import *
import librosa
import torch
from tqdm import tqdm
import re
from jiwer import wer
import jiwer
import sacrebleu
import jieba
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

normalize_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),  # if needed
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

class RetrieverInfer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm_type = cfg.get('llm_type')
        self.llm_path = cfg.get('llm_path')
        self.infer_task = cfg.get('infer_task')
        self.audio_lang = cfg.get('audio_lang')
        self.tgt_lang = cfg.get('audio_lang') if self.infer_task == 'asr' else cfg.get('tgt_lang')
        self.results_folder = cfg.get('results_folder')
        self.sample_rate = cfg.get('sample_rate')
        self.batch_size = cfg.get('batch_size')
        self.term_bank = cfg.get('term_bank_path')
        self.retriever_ans = cfg.get('retriever_ans')
        self.topk = cfg.get('topk')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.llm_type == 'qwen2-audio':
            self.llm = Qwen2AudioForConditionalGeneration.from_pretrained(self.llm_path, device_map="auto",torch_dtype=torch.bfloat16)
            self.processor = AutoProcessor.from_pretrained(self.llm_path)
            self.llm.eval()
            self.llm.to(device=self.device)
        elif self.llm_type == 'qwen-audio':
            self.processor = AutoTokenizer.from_pretrained(self.llm_path,trust_remote_code=True, device_map="auto")
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path,trust_remote_code=True, device_map="auto")
            self.llm.eval()
            self.llm.to(device=self.device)

        self.retriever = Retriever(cfg)
        self.retriever.to(torch.bfloat16)
        ckpt = torch.load(cfg.get('retriever_path'))['model']
        self.retriever.load_state_dict(ckpt,strict=True)

        self.retriever.eval()

        self.retriever.to(device=self.device)
        src_lang = 'English' if self.audio_lang == 'en' else 'Chinese'
        tgt_lang = 'English' if self.tgt_lang == 'en' else 'Chinese'
        self.st_prompt = 'You are an expert in speech translation.'
        self.asr_prompt = 'You are an expert in speech recognition.'
        self.prompt = self.asr_prompt if self.infer_task == 'asr' else self.st_prompt
        self.asr_text = f"This is an {src_lang} audio recording. Please transcribe this audio into {src_lang} text."
        self.st_text = f"This is an {src_lang} audio recording.  Please translate this audio into {tgt_lang} text."
        self.text = self.asr_text if self.infer_task == 'asr' else self.st_text
        self.term_bank = json.load(open(self.term_bank,'r'))
        if self.retriever_ans != None:
            self.retriever_ans = json.load(open(self.retriever_ans,'r'))

    def get_term_bank_embedding(self):
        if self.llm_type == 'qwen2-audio':
            return self.qwen2_text()
        elif self.llm_type == 'qwen-audio':
            return self.qwen_text()
        
    def get_audio_embedding(self, audio_file):
        if self.llm_type == 'qwen2-audio':
            return self.qwen2_audio(audio_file)
        elif self.llm_type == 'qwen-audio':
            return self.qwen_audio(audio_file)

    def qwen2_text(self):
        inputs = self.processor.tokenizer(self.term_bank[self.audio_lang], padding_side="right", add_special_tokens=False, return_tensors="pt",truncation=True, padding=True)
        attention_mask = inputs['attention_mask'].to(self.llm.device)
        input_ids = inputs['input_ids'].to(self.llm.device)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        return inputs_embeds, attention_mask
    
    def qwen2_audio(self, audio_file):
        audio_data, sr = librosa.load(audio_file,sr=self.sample_rate)
        audio_inputs = self.processor.feature_extractor(audio_data, sampling_rate=self.sample_rate)
        feature_attention_mask = audio_inputs['attention_mask']
        input_features = audio_inputs['input_features']
        input_features = torch.from_numpy(input_features)
        feature_attention_mask = torch.from_numpy(feature_attention_mask)
        audio_feat_lengths, audio_output_lengths = self.llm.audio_tower._get_feat_extract_output_lengths(
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
            dtype=self.llm.audio_tower.conv1.weight.dtype, device=self.llm.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        input_features = input_features.to(
            dtype=self.llm.audio_tower.conv1.weight.dtype, device=self.llm.audio_tower.conv1.weight.device
        )

        audio_outputs = self.llm.audio_tower(input_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.llm.multi_modal_projector(selected_audio_feature)
        audio_attention_mask = torch.ones((audio_features.shape[0],audio_features.shape[1])).to(audio_features.device)

        return audio_features, audio_attention_mask
    
    def qwen_audio(self, audio_file):
        audio_data = audio_file
        audio_data = self.processor.process_audio([audio_data])
        audio_features = self.llm.transformer.audio.encode(input_audios=audio_data['input_audios'], input_audio_lengths=audio_data['input_audio_lengths'], audio_span_tokens=audio_data['audio_span_tokens']).to(self.device)
        audio_attention_mask = torch.ones((audio_features.shape[0],audio_features.shape[1])).to(self.device)
        return audio_features, audio_attention_mask
    
    def qwen_text(self):
        src_term_bank = self.term_bank[self.audio_lang]
        inputs = [self.processor.convert_tokens_to_ids(self.processor.tokenize(src_term)) for src_term in src_term_bank]
        max_len = max([len(input) for input in inputs])
        input_ids = torch.full((len(inputs),max_len),self.processor.tokenizer.eot_token)
        for i in range(len(inputs)):
            now_length = len(inputs[i])
            input_ids[i,:now_length]=torch.tensor(inputs[i])
        attention_mask = (input_ids!=self.processor.tokenizer.eot_token).long().to(self.device)
        inputs_embeds = self.llm.transformer.wte(input_ids.to(self.device))

        return inputs_embeds, attention_mask
    
    def infer(self):
        test_datas = json.load(open(self.cfg.get('test_files'),'r'))
        time_words_embeds, time_word_ids_mask = self.get_term_bank_embedding() #size_term length embedding
        term_bank_src = self.term_bank[self.audio_lang]
        term_bank_trg = self.term_bank[self.tgt_lang]

        infer_datas = []
        recalls = []
        for test_data in tqdm(test_datas):
            audio_file = test_data['wav_path']
            audio_embedding, audio_attention_mask = self.get_audio_embedding(audio_file)
            avg_lprobs = self.retriever(time_words_embeds=time_words_embeds, time_word_ids_mask=time_word_ids_mask, audio_embeds=audio_embedding, audio_embeds_mask=audio_attention_mask)
            values_at_dim2_index0 = avg_lprobs[:, :, 1]
            # topK
            top_10_values, top_10_indices = torch.topk(values_at_dim2_index0, self.topk, dim=1, largest=True)

            retriever_src_str = []
            retriever_trg_str = []
            keywords_pair = []

            for i in range(self.topk):
                retriever_src_str.append(term_bank_src[top_10_indices[0][i]])
                retriever_trg_str.append(term_bank_trg[top_10_indices[0][i]])
                keywords_pair.append(term_bank_src[top_10_indices[0][i]]+':'+term_bank_trg[top_10_indices[0][i]])

            asr_gt_term = [hint[self.audio_lang] for hint in test_data['hints']]
            st_gt_term = [hint[self.tgt_lang] for hint in test_data['hints']]
            asr_intersection = list(set(asr_gt_term) & set(retriever_src_str))
            st_intersection = list(set(st_gt_term) & set(retriever_trg_str))

            recall = len(asr_intersection) / len(asr_gt_term)

            infer_data = {
                'id':test_data['wav_path'],
                'retriever_src_str':retriever_src_str,
                'retriever_trg_str':retriever_trg_str,
                'asr_gt_term': asr_gt_term,
                'st_gt_term': st_gt_term,
                'recall':recall
            }
            infer_datas.append(infer_data)
            recalls.append(recall)

        print(sum(recalls)/len(recalls))
        json.dump(infer_datas,open(self.results_folder+'/infer_results.json','w'),ensure_ascii=False,indent=4)

    def infer_llm(self):
        test_datas = json.load(open(self.cfg.get('test_files'),'r'))
        infer_datas = []
        precisions = []
        count = 0
        for test_data in tqdm(test_datas):
            if self.retriever_ans != None:
                assert test_data['wav_path'] == self.retriever_ans[count]['id']
                if self.infer_task == 'asr':
                    retriever_ans ="Specialized terminology may appear in the audio. Please accurately recognize these terms. Potential technical terms include:"+', '.join(self.retriever_ans[count]['retriever_src_str'])
                else:
                    retriever_ans = "Specialized terminology may appear in the audio. Please accurately translate these terms. Potential technical terms include:"+', '.join(self.retriever_ans[count]['retriever_trg_str'])
                count+=1
            else:
                retriever_ans = ""

            text_with_retriever = self.text + retriever_ans

            conversation = [
                {'role': 'system', 'content': self.prompt}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": test_data['wav_path']},
                    {"type": "text", "text": text_with_retriever}
                ]}
            ]

            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audio_file = test_data['wav_path']
            audio, sr = librosa.load(audio_file, sr=self.processor.feature_extractor.sampling_rate)
            inputs = self.processor(text=text, audio=[audio], return_tensors="pt",sampling_rate=self.processor.feature_extractor.sampling_rate).to(self.llm.device)
            
            generated_ids = self.llm.generate(**inputs, max_new_tokens=256)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if self.infer_task == 'asr':
                term_lang = self.audio_lang
            else:
                term_lang = self.tgt_lang
            terms = [(1 if term[term_lang] in output_text else 0) for term in test_data['hints']]
            precision = sum(terms) / len(terms)
            hyp_norm = output_text
            tgt_norm = test_data[self.tgt_lang]

            if self.infer_task == 'asr':
                hyp_norm = normalize_transform(hyp_norm)
                tgt_norm = normalize_transform(tgt_norm)
                if self.tgt_lang == 'zh':
                    criterion = jiwer.cer(tgt_norm, hyp_norm)
                elif self.tgt_lang == 'en':
                    criterion = jiwer.wer(tgt_norm, hyp_norm)

            if self.infer_task == 'st':
                if self.tgt_lang == 'zh':
                    criterion = sacrebleu.corpus_bleu([hyp_norm], [[tgt_norm]], tokenize=self.tgt_lang).score
                else:
                    criterion = sacrebleu.corpus_bleu([hyp_norm], [[tgt_norm]]).score

            infer_data = {
                'id':test_data['wav_path'],
                'hyp':output_text,
                'ref':test_data[self.tgt_lang],
                'ref_hints':test_data['hints'],
                'precision':precision,
                'criterion':criterion 
            }
            infer_datas.append(infer_data)

        json.dump(infer_datas,open(self.results_folder+'/infer_results.json','w'),ensure_ascii=False,indent=4)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--infer-class', type=str, help='retriever,llm')
    args = parser.parse_args()
    print(args)
    with open(args.config) as f:
        cfg = json.load(f)
    
    retriever_infer = RetrieverInfer(cfg)

    if args.infer_class == 'retriever':
        retriever_infer.infer()
    elif args.infer_class == 'llm':
        retriever_infer.infer_llm()