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
    TransformerModelBaseRerank,
)
from fairseq.models.transformer.transformer_base_rerank import GeneratorHubInterfaceCandidates
from torch import Tensor
import logging
import copy, os
import numpy as np
from fairseq.dataclass.utils import overwrite_args_by_name
from fairseq.data import encoders
import math
from torch.nn import Parameter
from omegaconf import open_dict


logger = logging.getLogger(__name__)


class TransformerModelBaseTwist(TransformerModelBaseRerank):
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
        tgt_cfg = copy.deepcopy(x["args"])
        for key, val in kwargs.items():
            if key[:4] == 'tgt_':
                kwargs[key[4:]] = val
        overwrite_args_by_name(tgt_cfg, kwargs)
        # Modify so that we get candidates
        return GeneratorHubInterfaceTwist(x["args"], tgt_cfg, x["task"], x["models"], model_name_or_path, checkpoint_file)


class GeneratorHubInterfaceTwist(GeneratorHubInterfaceCandidates):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """
    def __init__(self, cfg, tgt_cfg, task, models, model_name_or_path, checkpoint_file):
        super().__init__(cfg, tgt_cfg, task, models)
        #self.distances = self.load_distances(model_name_or_path, checkpoint_file)
        # [v, v] distance matrix

    def load_distances(self, model_name_or_path, checkpoint_file):
        with open(os.path.join(model_name_or_path, checkpoint_file.replace('.pt', '_') + 'distances.npy'), "rb") as f:
        #with open(os.path.join(model_name_or_path, checkpoint_file.replace('.pt', '_') + 'normalized-distances.npy'), "rb") as f:
        #with open(os.path.join(model_name_or_path, checkpoint_file.replace('.pt', '_') + 'distances_orig.npy'), "rb") as f:
            distances = np.load(f)
        distances = Parameter(torch.tensor(distances), requires_grad=False)
        return distances

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, candidates: List[List[str]] = None, lmd: float = 1.0, encoder_outs=None, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, candidates=candidates, lmd=lmd, encoder_outs=encoder_outs, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        r2l=kwargs.get("r2l", False)
        kwargs["twist"] = True
        inference_step_args = {}
        inference_step_args["lmd"] = lmd
        #inference_step_args["distances"] = self.distances
        if candidates is not None:
            candidates = [[self.encode(candidate, target=True, r2l=r2l) for candidate in candidate_sent] for candidate_sent in candidates]
            inference_step_args["candidates"] = candidates
        #test_out = self.generate(tokenized_sentences, beam, verbose, inference_step_args=inference_step_args, encoder_outs=encoder_outs, **kwargs)
        #batched_hypos, encoder_outs = test_out[0], test_out[1]
        batched_hypos, encoder_outs = self.generate(tokenized_sentences, beam, verbose, inference_step_args=inference_step_args, encoder_outs=encoder_outs, **kwargs)
        return [[self.decode(hypo["tokens"], r2l=r2l) for hypo in hypos] for hypos in batched_hypos], batched_hypos, encoder_outs

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        prefix_allowed_tokens_fn=None,
        encoder_outs=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            #prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        inference_step_args = inference_step_args or {}
        results = []
        encoder_results = []
        batch_idx = 0
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            if encoder_outs is not None:
                encoder_outs_batch = encoder_outs[batch_idx]
            else:
                encoder_outs_batch = None
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args, encoder_outs=encoder_outs_batch,
            )
            if isinstance(translations, tuple):
                translations, encoder_outs_batch = translations[0], translations[1]
            #else:
            #    translations = self.task.inference_step(
            #        generator, self.models, batch, **inference_step_args, encoder_outs=encoder_outs_batch,
            #    )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))
            encoder_results.append(encoder_outs_batch)
            batch_idx += 1

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )

        return outputs, encoder_results
