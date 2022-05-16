# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import sys
from copy import deepcopy

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel

# TODO: For now, assume the same vocabulary. Later relax this assumption.
class SequenceRerankSameVocabGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """

        assert len(models) > 1, "The last model is the reranker. We thus need 2+ models."
        super().__init__(
            models[:-1],
            tgt_dict,
            beam_size,
            max_len_a,
            max_len_b,
            max_len,
            min_len,
            normalize_scores,
            len_penalty,
            unk_penalty,
            temperature,
            match_source_len,
            no_repeat_ngram_size,
            search_strategy,
            eos,
            symbols_to_strip_from_output,
            lm_model,
            lm_weight,
        )
        self.reranker = models[-1]

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        finalized = super()._generate(
                    sample,
                    prefix_tokens,
                    constraints,
                    bos_token,
                    )
        finalized = self.rerank(finalized, sample)
        return finalized

    def rerank(self, finalized, sample):
        def rebuild_batch(finalized):
            finalized_tokens = [f["tokens"] for f_sent in finalized for f in f_sent]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0]
                .new_zeros(len(finalized_tokens), finalized_maxlen)
                .fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        encoder_out = self.reranker.encoder.forward_torchscript(net_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.reranker.encoder.reorder_encoder_out(encoder_out, new_order)
        prev_output_tokens = torch.zeros_like(final_output_tokens).fill_(self.pad)
        prev_output_tokens[:, 1:] = final_output_tokens[:, :-1]
        prev_output_tokens[prev_output_tokens.eq(self.eos)] = self.pad
        # Remove eos in prev_output_tokens
        prev_output_tokens[:, 0] = self.eos
        reranking_scores = self.reranker.decoder(prev_output_tokens, encoder_out)
        # Apply penalties
        reranking_scores[0][:, :, self.pad] = -math.inf  # never select pad
        reranking_scores[0][:, :, self.unk] -= self.unk_penalty  # apply unk penalty
        reranking_scores = self.reranker.get_normalized_probs(reranking_scores, True, None)
        reranking_scores = reranking_scores.gather(2, final_output_tokens.unsqueeze(-1)).squeeze(-1)
        reranking_mask = final_output_tokens.ne(self.pad)
        reranking_scores = reranking_scores.masked_fill_(~reranking_mask, 0)
        agg_reranking_scores = reranking_scores.sum(1)
        # [bsz*beam_size]
        if self.normalize_scores:
             # normalize sentence-level scores
            agg_reranking_scores /= (reranking_mask.sum(1) ** self.len_penalty)
        # Update scores
        for i in range(bsz):
            for j in range(beam_size):
                finalized[i][j]["score"] = agg_reranking_scores[i*self.beam_size+j]
        # Resort. A bit redundant but fine for now.
        for i in range(bsz):
            scores = torch.stack([finalized[i][j]["score"] for j in range(beam_size)])
            _, indices = scores.sort(descending=True)
            finalized_sent = [finalized[i][k] for k in indices]
            finalized[i] = finalized_sent

        return finalized
