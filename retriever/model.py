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


import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
import argparse
import math
EPSILON = 1e-5

class Xentropy(nn.Module):

    def __init__(self, label_smooth_factor=0.1):
        super().__init__()
        self.label_smooth_factor = label_smooth_factor

    def forward(self, logits, src_mask, target, target_mask, *_args, **_kwargs):
        """
        Calculating different loss for the training of the model.

        Args:

            - logits: [B, T, N], for N classes
            - src_mask: [B, Feature_T]
            - target: [B, T, N]
            - target_mask: [B, T]

        Return:

            - losses: a dict of variant loss functions
            - weight: the weight of all losses,\
            here we refer to the sum of effective labels in target batch.

        """

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        hyp_tgt = lprobs.max(dim=-1)[1]
        hyp_acc = ((hyp_tgt == target).float() * (target_mask.float())).sum() / (target_mask.float().sum() + EPSILON)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        smooth_loss = -lprobs.sum(dim=-1, keepdim=False)

        nll_loss = nll_loss.squeeze(-1)
        eps_i = self.label_smooth_factor / lprobs.size(-1)
        loss = (1.0 - self.label_smooth_factor) * nll_loss + eps_i * smooth_loss

        masked_loss = (loss * target_mask.float()).sum() / (target_mask.float().sum() + EPSILON)
        masked_nll_loss = (nll_loss * target_mask.float()).sum() / (target_mask.float().sum() + EPSILON)
        # for PPL
        sent_lprobs = lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        masked_lprobs = (sent_lprobs.squeeze(-1) * target_mask.float()).sum(dim=-1)  # sentence prob

        frame_size = src_mask.float().sum()
        tgt_size = target_mask.float().sum()  # tokens
        forward_out = {}
        forward_out['utt_num'] = src_mask.shape[0]
        forward_out['backward_loss'] = masked_loss
        forward_out['loss'] = masked_loss.type_as(logits)
        forward_out['lprobs'] = masked_lprobs.type_as(logits) 
        forward_out['nll_loss'] = masked_nll_loss.type_as(logits)
        forward_out['acc'] = hyp_acc
        forward_out['frame_size'] = frame_size
        forward_out['tgt_size'] = tgt_size

        return forward_out


class TorchMultiheadAttention(nn.MultiheadAttention):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype
        )
        self.merge_qkv = self._qkv_same_embed_dim
        self.bias = bias

class AudioWordAttn(nn.Module):
    '''AudioWordAttn'''

    def __init__(self, args, use_torch_mha=True):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        if not use_torch_mha:
            self.attn = MultiheadAttention(
                embed_dim=args.get("n_embed"),
                num_heads=args.get("retriever_n_head"),
                kdim=args.get("n_embed"),
                vdim=args.get("n_embed"),
                dropout=args.get("attn_pdrop"),
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
            )
        else:
            self.attn = TorchMultiheadAttention(
                embed_dim=args.get("n_embed"),
                num_heads=args.get("retriever_n_head"),
                kdim=args.get("n_embed"),
                vdim=args.get("n_embed"),
                dropout=args.get("attn_pdrop"),
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
            )

    def forward(self, query1, value, key_padding_mask=None, query2=None, need_weights=True):
        '''query1: [B, U, D]   ---- (B,N,L,D),
        query2: [B, U, D']
        value: [N, D] or [B, N, D]   ----(B,T,D)
        output: [B, U, D]
        '''
        if query2 is None:
            query = query1
        bsz = query.size(0)
        if len(value.size()) == 2:
            value = value.unsqueeze(0).repeat(bsz, 1, 1)
        if len(query.size()) == 4:
            query = torch.flatten(query, start_dim=1, end_dim=2)  # (B, N*L, D)
        if key_padding_mask is not None:
            key_padding_mask = (1 - key_padding_mask).bool()
        v = value.permute(1, 0, 2)
        attn_out, attn_weights = self.attn(
            query=query.permute(1, 0, 2),
            key=v,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return attn_out.permute(1, 0, 2), attn_weights


class Retriever(nn.Module):
    '''Binaty CE Retriever'''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.audio_word_att = AudioWordAttn(args)
        self.retriever_merge_type = args.get('retriever_merge_type', 'fc')
        self.retriever_audio_word_fc = args.get('retriever_audio_word_fc', True)
        self.time_words_proj = nn.Linear(args.get("n_embed"), args.get("n_embed"))  # 5120,2560
        if self.retriever_audio_word_fc:
            if self.retriever_merge_type == 'add':
                self.audio_word_fc = nn.Linear(args.get("n_embed"), args.get("n_embed"), bias=True)
            else:
                self.audio_word_fc = nn.Linear(args.get("n_embed") * 2, args.get("n_embed"), bias=True)
        self.audio_word_proj = nn.Linear(args.get("n_embed"), 2)

        self.retiever_criterion = Xentropy()

    def forward(
        self, time_words_embeds, time_word_ids_mask, audio_embeds, audio_embeds_mask, is_inference=True, label=None
    ):
        # audio_embeds(batch_audio, audio_length, audio_embed_dim)
        # time_words_embeds(size of term, time_word_ids_length,llm_embed_dim)
        #label(batch_audio,size of term) 1.0 means positive 0.0 means negative
        bsz = audio_embeds.shape[0]
        time_words_embeds = self.time_words_proj(time_words_embeds)
        time_words_embeds = time_words_embeds.unsqueeze(0).repeat(bsz, 1, 1, 1)
            
        # au_att
        audio_embeds_out, _ = self.audio_word_att(
            time_words_embeds,
            audio_embeds,
            key_padding_mask=audio_embeds_mask,
            need_weights=False,
        )  # (B,N,L,D),(B,T,D)   -> (B,N*L,D)

        # au_att_pooling
        audio_embeds_out_masked = audio_embeds_out * torch.flatten(
            time_word_ids_mask, start_dim=0, end_dim=1
        ).unsqueeze(0).unsqueeze(
            2
        )  # (B,N*L,D)
        audio_embeds_out_masked = audio_embeds_out_masked.view(
            bsz, time_words_embeds.shape[1], time_words_embeds.shape[2], time_words_embeds.shape[3]
        )
        audio_embeds_out_masked_sum = torch.sum(audio_embeds_out_masked, dim=2)  # (B,N,D)
        time_word_ids_mask_sum = torch.sum(time_word_ids_mask, dim=1)  # (N)
        audio_embeds_out = (
            (audio_embeds_out_masked_sum / time_word_ids_mask_sum.unsqueeze(0).unsqueeze(2))
            .type_as(audio_embeds)
            .to(audio_embeds.device)
        )  # (B,N,D)
        audio_embeds_out = audio_embeds_out.unsqueeze(2).repeat(1, 1, time_words_embeds.shape[2], 1)
        # retriever methods
        if self.retriever_merge_type == 'add':
            audio_word_concat = torch.add(audio_embeds_out, time_words_embeds)
        else:
            audio_word_concat = torch.cat((audio_embeds_out, time_words_embeds), dim=3)  # (B,N,L,D, BNLD -> BNLD)
        if self.retriever_audio_word_fc:
            audio_word_concat_fc = self.audio_word_fc(audio_word_concat)
        else:
            audio_word_concat_fc = audio_word_concat
        audio_word_concat_proj = self.audio_word_proj(audio_word_concat_fc)  # BNLD
        audio_word_concat_proj_masked = audio_word_concat_proj * time_word_ids_mask.unsqueeze(0).repeat(
            bsz, 1, 1
        ).unsqueeze(3)
        audio_word_concat_proj_sum = torch.sum(audio_word_concat_proj_masked, dim=2)  # (B,N,D)
        time_word_ids_mask_sum = torch.sum(time_word_ids_mask, dim=1)  # (N)
        audio_word_concat_avg_proj = (
            (audio_word_concat_proj_sum / time_word_ids_mask_sum.unsqueeze(0).unsqueeze(2))
            .type_as(audio_embeds)
            .to(audio_embeds.device)
        )  # (B,N,D)
        # audioB, time_wordB,2
        if not is_inference:

            time_word_ids_match_label_mask = torch.ones(bsz, label.shape[1], dtype=torch.float32).to(
                audio_word_concat_avg_proj.device
            )
            audio_embeds_out_mask = torch.ones(bsz, audio_embeds_out.shape[1], dtype=torch.float32).to(
                audio_word_concat_avg_proj.device
            )
            bce_avg_out = self.retiever_criterion(
                audio_word_concat_avg_proj, audio_embeds_out_mask, label, time_word_ids_match_label_mask
            )
            return bce_avg_out
        avg_lprobs = F.log_softmax(audio_word_concat_avg_proj, dim=-1, dtype=torch.float32)

        return avg_lprobs


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--n_embed', type=int, default=1024)
    args.add_argument('--n_head', type=int, default=8)
    args.add_argument('--attn_pdrop', type=float, default=0.1)
    args=args.parse_args()
    model=Retriever(args)
    time_words_embeds=torch.rand(2,3,1024)
    time_word_ids_mask=torch.rand(2,3)
    audio_embeds=torch.rand(11,7,1024)
    audio_embeds_mask=torch.rand(11,7)
    ouputs = model(time_words_embeds, time_word_ids_mask, audio_embeds, audio_embeds_mask)
    pass