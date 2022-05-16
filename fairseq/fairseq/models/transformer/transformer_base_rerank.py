# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerModelBase,
)
from fairseq.hub_utils import GeneratorHubInterface
from torch import Tensor
import logging
from copy import deepcopy
from fairseq.dataclass.utils import overwrite_args_by_name
from fairseq.data import encoders
import math
from fairseq.tokenizer import tokenize_line
logger = logging.getLogger(__name__)


class TransformerModelBaseRerank(TransformerModelBase):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        logger.info(x["args"])
        # fairseq doesn't support tokenizer/bpe for the target.
        # We create a copy to get the target side.
        # Default: tokenize/bpe with src; detokenize/debpe with tgt.
        tgt_cfg = deepcopy(x["args"])
        for key, val in kwargs.items():
            if key[:4] == 'tgt_':
                kwargs[key[4:]] = val
        overwrite_args_by_name(tgt_cfg, kwargs)
        # Modify so that we get candidates
        return GeneratorHubInterfaceCandidates(x["args"], tgt_cfg, x["task"], x["models"])

class GeneratorHubInterfaceCandidates(GeneratorHubInterface):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """
    def __init__(self, cfg, tgt_cfg, task, models):
        super().__init__(cfg, task, models)
        tgt_dict = task.target_dictionary
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        # Only support solo models for now.
        #assert len(models) == 1
        self.reranker = models[0]
        self.tgt_tokenizer = encoders.build_tokenizer(tgt_cfg.tokenizer)
        self.tgt_bpe = encoders.build_bpe(tgt_cfg.bpe)

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        langcode=kwargs.get("langcode", None)
        r2l=kwargs.get("r2l", False)
        candidates=kwargs.get("candidates", None)
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        if candidates is not None:
            # Candidates are already given. Do not generate.
            # Make sure we use target tokenizer/bpe
            candidates = [[self.encode(candidate, target=True, r2l=r2l) for candidate in candidate_sent] for candidate_sent in candidates]
            batched_hypos = self.rerank(tokenized_sentences, candidates, **kwargs)
        else:
            batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        # Generate all candidates in addition to the best.
        return [[self.decode(hypo["tokens"], r2l=r2l) for hypo in hypos] for hypos in batched_hypos], batched_hypos

    def rerank(self, tokenized_sentences, finalized, **kwargs):
        normalize_scores=(not getattr(kwargs, "unnormalized", False))
        len_penalty=kwargs.get("lenpen", 1)
        unk_penalty=kwargs.get("unkpen", 0)
        langcode=kwargs.get("langcode", None)
        beam_size = len(finalized[0])
        src_tokens = self.rebuild_batch(tokenized_sentences, left_padding=True, langcode=langcode)
        finalized_output_tokens = [f for f_sent in finalized for f in f_sent]
        final_output_tokens = self.rebuild_batch(finalized_output_tokens, langcode=langcode)
        net_input = {}
        net_input["src_tokens"] = src_tokens
        bsz = src_tokens.shape[0]
        encoder_out = self.reranker.encoder.forward_torchscript(net_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.reranker.encoder.reorder_encoder_out(encoder_out, new_order)
        prev_output_tokens = torch.zeros_like(final_output_tokens).fill_(self.pad)
        prev_output_tokens[:, 1:] = final_output_tokens[:, :-1]
        # Remove eos in prev_output_tokens
        prev_output_tokens[prev_output_tokens.eq(self.eos)] = self.pad
        # Add eos to the beginning
        prev_output_tokens[:, 0] = self.eos
        reranking_scores = self.reranker.decoder(prev_output_tokens, encoder_out)
        # Apply penalties
        reranking_scores[0][:, :, self.pad] = -math.inf  # never select pad
        reranking_scores[0][:, :, self.unk] -= unk_penalty  # apply unk penalty
        reranking_scores = self.reranker.get_normalized_probs(reranking_scores, True, None)
        reranking_scores = reranking_scores.gather(2, final_output_tokens.unsqueeze(-1)).squeeze(-1)
        if langcode is not None:
            final_output_tokens = final_output_tokens[:, 1:]
            reranking_scores = reranking_scores[:, 1:]
        reranking_mask = final_output_tokens.ne(self.pad)
        reranking_scores = reranking_scores.masked_fill_(~reranking_mask, 0)
        agg_reranking_scores = reranking_scores.sum(1)
        # [bsz*beam_size]
        if normalize_scores:
            # normalize sentence-level scores
            agg_reranking_scores /= (reranking_mask.sum(1) ** len_penalty)
            agg_reranking_scores = agg_reranking_scores.view(bsz, beam_size)
        # Sort finalized
        _, indices = agg_reranking_scores.sort(descending=True, dim=1)
        batched_hypos = []
        for sent in range(bsz):
            hypos = []
            indices_sent = indices[sent]
            for idx in indices_sent:
                hypo = {}
                hypo["tokens"] = finalized[sent][idx]
                hypo["score"] = agg_reranking_scores[sent, idx]
                reranking_scores_hypo = reranking_scores.view(bsz, beam_size, -1)[sent, idx]
                reranking_mask_hypo = reranking_mask.view(bsz, beam_size, -1)[sent, idx]
                hypo["positional_scores"] = reranking_scores_hypo[reranking_mask_hypo]
                hypos.append(hypo)
            batched_hypos.append(hypos)
        return  batched_hypos

    def encode(self, sentence: str, target: bool = False, r2l: bool = False) -> torch.LongTensor:
        sentence = self.tokenize(sentence, target)
        sentence = self.apply_bpe(sentence, target)
        if r2l:
            sentence = tokenize_line(sentence)
            sentence = ' '.join(sentence[::-1])
        return self.binarize(sentence, target)

    def tokenize(self, sentence: str, target: bool = False) -> str:
        if target and (self.tgt_tokenizer is not None):
            sentence = self.tgt_tokenizer.encode(sentence)
        elif self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def apply_bpe(self, sentence: str, target: bool = False) -> str:
        if target and (self.tgt_bpe is not None):
            sentence = self.tgt_bpe.encode(sentence)
        elif self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: str, target: bool = False) -> str:
        if target and (self.tgt_bpe is not None):
            sentence = self.tgt_bpe.decode(sentence)
        elif self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: str, target: bool = False) -> torch.LongTensor:
        if target:
            return self.tgt_dict.encode_line(sentence, add_if_not_exist=False).long()
        return self.src_dict.encode_line(sentence, add_if_not_exist=False).long()

    def decode(self, tokens: torch.LongTensor, target: bool = True, r2l: bool = False) -> str:
        sentence = self.string(tokens, target)
        if r2l:
            sentence = tokenize_line(sentence)
            sentence = ' '.join(sentence[::-1])
        sentence = self.remove_bpe(sentence, target)
        return self.detokenize(sentence)

    def detokenize(self, sentence: str, target: bool = True) -> str:
        if target and (self.tgt_tokenizer is not None):
            sentence = self.tgt_tokenizer.decode(sentence)
        elif self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def string(self, tokens: torch.LongTensor, target: bool = True) -> str:
        if target:
            return self.tgt_dict.string(tokens)
        return self.src_dict.string(tokens)

    def rebuild_batch(self, finalized_tokens, left_padding=False, langcode=None):
        finalized_maxlen = max(f.size(0) for f in finalized_tokens)
        if langcode is not None:
            finalized_maxlen += 1
        final_output_tokens = (
            finalized_tokens[0]
            .new_zeros(len(finalized_tokens), finalized_maxlen)
            .fill_(self.pad)
        )
        for i, f in enumerate(finalized_tokens):
            if langcode is not None:
                f = torch.cat([torch.LongTensor([self.tgt_dict.index(langcode)]), f])
            if left_padding: 
                final_output_tokens[i, -f.size(0):] = f
            else:
                final_output_tokens[i, : f.size(0)] = f
        final_output_tokens = utils.apply_to_sample(lambda t: t.to(self.device), final_output_tokens)
        return final_output_tokens
