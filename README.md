# Twist Decoding: Diverse Generators Guide Each Other

<p align="center">
<img src="https://github.com/jungokasai/twist_decoding/blob/main/figs/twist_decoding.png" height="300" alt="twist_decoding">
</p>

## Introduction
Many language generation models have different settings, such as vocabularies, tokenization, and generation order, so they can't be simply ensembled. Our Twist decoding combines models regardless of such differences without any additional training or finetuning.

## Installation
We forked the [fairseq library](https://github.com/pytorch/fairseq) and incorporated [distance terms](https://github.com/jungokasai/twist_decoding/blob/main/fairseq/fairseq/sequence_twist_generator.py#L236) to their beam implementation.
You can incorporate this in any implementation of beam search, but here we provide the codebase that we used for our paper.
To run experiments, follow the [fairseq](https://github.com/pytorch/fairseq) instructions and run in this repository:
```bash
cd fairseq
pip install --editable .
python setup.py build_ext --inplace
```
## Download Our Models and Data
Any fairseq seq-to-seq model should work, but here we provide all models we used in our experiments. See our [paper]() for the training details.
| Models | | | | |
|---|---|---|---|---|
[DE-EN Generic](https://arkdata.cs.washington.edu/twist/domains/wmt19.de-en.joined-dict.tar.gz)<sup>[1](#footnote1)</sup> | [DE-EN Medicine](https://arkdata.cs.washington.edu/twist/domains/trans-base_medicine-de-en.tar.gz) | [DE-EN Law](https://arkdata.cs.washington.edu/twist/domains/trans-base_law-de-en.tar.gz) | [DE-EN Koran](https://arkdata.cs.washington.edu/twist/domains/trans-base_koran-de-en.tar.gz)  | [DE-EN Subtitles](https://arkdata.cs.washington.edu/twist/domains/trans-base_subtitles-de-en.tar.gz) |
[ZH-EN L2R](https://arkdata.cs.washington.edu/twist/l2r-r2l/trans-large-l2r_wmt20-zh-en.tar.gz) | [ZH-EN R2L](https://arkdata.cs.washington.edu/twist/l2r-r2l/trans-large-r2l_wmt20-zh-en.tar.gz)| [EN-DE L2R](https://arkdata.cs.washington.edu/twist/l2r-r2l/trans-large-l2r_wmt20-en-de.tar.gz) | [EN-DE R2L](https://arkdata.cs.washington.edu/twist/l2r-r2l/trans-large-r2l_wmt20-en-de.tar.gz) | |
[SciTLDR Abstract](https://arkdata.cs.washington.edu/twist/scitldr/scitldr_bart.tldr-ao.tar.gz)<sup>[2](#footnote2)</sup> | [SciTLDR AIC](https://arkdata.cs.washington.edu/twist/scitldr/scitldr_catts-xsum.tldr-aic.tar.gz)<sup>[2](#footnote2)</sup>  | | | |

<a name="footnote1">1</a>: WMT19 top-performing model. Downloaded from the [fairseq repository](https://github.com/pytorch/fairseq/tree/main/examples/wmt19).</br>
<a name="footnote2">2</a>: Downloaded from the [official repository](https://github.com/allenai/scitldr) of the SciTLDR dataset ([Cachola et al., 2020](https://arxiv.org/abs/2004.15011)).


| Datasets | | | | | |
|---|---|---|---|---|---|
[DE-EN Medicine](https://arkdata.cs.washington.edu/twist/domains/medicine-de-en_data.tar.gz)<sup>[3](#footnote3)</sup> | [DE-EN Law](https://arkdata.cs.washington.edu/twist/domains/law-de-en_data.tar.gz)<sup>[3](#footnote3)</sup> | [DE-EN Koran](https://arkdata.cs.washington.edu/twist/domains/koran-de-en_data.tar.gz)<sup>[3](#footnote3)</sup> | [EN-DE Subtitles](https://arkdata.cs.washington.edu/twist/domains/subtitles-de-en_data.tar.gz)<sup>[3](#footnote3)</sup> | [WMT20 ZH-EN](https://arkdata.cs.washington.edu/billboard/wmt20-zh-en/data/wmt20-zh-en_bpe32k.tar.gz)<sup>[4](#footnote4)</sup> | [WMT20 EN-DE](https://arkdata.cs.washington.edu/billboard/wmt20-en-de/data/wmt20-en-de_bpe32k.tar.gz)<sup>[4](#footnote4)</sup> |

<a name="footnote3">3</a>:  Downloaded from the [official repository](https://github.com/JunjieHu/dali) of [Hu et al. (2019)](https://arxiv.org/abs/1906.00376).</br>
<a name="footnote4">4</a>:  Downloaded from the [official repository](https://github.com/jungokasai/billboard/tree/master/baselines) of the bidimensional leaderboards ([Kasai et al., 2022](https://arxiv.org/abs/2112.04139)).

## Decode Domain and Generic Models
Here are some example commands.
Run Twist decoding with `f=Domain` and `g=Generic` in the medical domain.
They are separated by a colon in options: `f:g`.
Run [Moses](https://github.com/moses-smt/mosesdecoder) detokenization after.
```bash
cd fairseq/
python twist/generate_twist.py --model-dirs  <PATH>/trans-base_medicine-de-en/:<PATH>/wmt19.de-en.joined-dict/ --model-names model.pt:model.pt --out-file mt/domains/medicine/output/test.twist --r2l 0:0 --src-lang de --tgt-lang en --in-file mt/domains/medicine/src/emea-test.tok.de --batch-size 20 --max-updates 3 --lmd-g 0.3 --lmd-f 0.1
perl <PATH>/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < mt/domains/medicine/output/test.twist_update-2.out > mt/domains/medicine/output/test.twist_update-2.txt
```
Run Twist decoding with `f=Generic` and `g=Domain` in the legal domain.
```bash
python twist/generate_twist.py --model-dirs  <PATH>/wmt19.de-en.joined-dict/:<PATH>/trans-base_law-de-en/ --model-names model.pt:model.pt --out-file mt/domains/law/output/test.twist --r2l 0:0 --src-lang de --tgt-lang en --in-file mt/domains/law/src/acquis-test.tok.de --batch-size 20 --max-updates 3 --lmd-g 3.0 --lmd-f 0.1
```
Run the reranking baseline.
```bash
python twist/generate_rerank.py --model-dirs  <PATH>/trans-base_medicine-de-en/:<PATH>/wmt19.de-en.joined-dict/ --model-names model.pt:model.pt --out-file mt/domains/medicine/output/test.rerank.out --r2l 0:0 --src-lang de --tgt-lang en --in-file mt/domains/medicine/src/emea-test.tok.de --batch-size 20
```
## Decode Left-to-Right and Right-to-Left Models 
The command is similar, but we pass the `--r2l` option.
```bash
python twist/generate_twist.py  --model-dirs <PATH>/trans-large-r2l_wmt20-zh-en/:<PATH>/trans-large-l2r_wmt20-zh-en/ --model-names model.pt:model.pt --out-file mt/wmt/zh-en/output/test.twist --r2l 1:0 --src-lang zh --tgt-lang en --in-file mt/wmt/zh-en/src/newstest2020.zh-en.src.tok.zh --max-updates 3 --lmd-g 3.0 --lmd-f 0.1 --batch-size 20
```

## Paper Summarization
Here are some example commands.
Run Twist decoding with `f=AIC` (abstract, introduction, and conclusion) and `g=Abstract`. 
```bash
python twist/generate_twist_tldr.py --checkpoint-dirs <PATH>/scitldr_catts-xsum.tldr-aic/:<PATH>/scitldr_bart.tldr-ao/ --data-dirs summ/scitldr/SciTLDR-AIC/ctrl:summ/scitldr/SciTLDR-A/ctrl --checkpoint-files scitldr_catts-xsum.tldr-aic.pt:scitldr_bart.tldr-ao.pt --max-updates 3 --batch-size 1 --split test --beam 5 --lmd-g 3.0 --lmd-f 0.3 --batch-size 1 --out-file summ/scitldr/output/test.twist
```
Run the reranking baseline.
```bash
python twist/generate_rerank_tldr.py --checkpoint-dirs <PATH>/scitldr_catts-xsum.tldr-aic/:<PATH>/scitldr_bart.tldr-ao --data-dirs summ/scitldr/SciTLDR-AIC/ctrl:summ/scitldr/SciTLDR-A/ctrl --checkpoint-files scitldr_catts-xsum.tldr-aic.pt:scitldr_bart.tldr-ao.pt --batch-size 1 --split test --beam 5 --batch-size 1 --out-file summ/scitldr/output/test.rerank.txt
```

## Evaluate Results
Lastly, we provide tools for evaluations: [COMET](https://aclanthology.org/2020.wmt-1.101/) for machine translation and [ROUGE](https://aclanthology.org/W04-1013/) for summarization.
Use the [sacrebleu](https://github.com/mjpost/sacrebleu) library to measure the BLEU score.
For example,
```bash
cd eval/COMET/
bash run.sh  ../../fairseq/mt/domains/medicine/src/emea-test.de ../../fairseq/mt/domains/medicine/output/test.twist_update-2.txt ../../fairseq/mt/domains/medicine/tgt/emea-test.en.jsonl ../../fairseq/mt/domains/medicine/output/test.twist_update-2.comet
cd fairseq/
sacrebleu mt/domains/medicine/tgt/emea-test.en -i mt/domains/medicine/output/test.twist_update-2.txt -m bleu -b -w 4 -l de-en
```
```bash
cd eval/ROUGE/
bash run.sh  ../../fairseq/summ/scitldr/output/test.twist_update-2.txt  ../../fairseq/summ/scitldr/output/test.twist_update-2.txt    ../../fairseq/summ/scitldr/tgt/test_refs.jsonl   ../../fairseq/summ/scitldr/output/test.twist_update-2.rougeL rougeL
```

## Citation
```
@misc{kasai2022twist,
  author    = {Jungo Kasai and
               Keisuke Sakaguchi and
               Ronan Le Bras and
               Hao Peng and
               Ximing Lu and
               Dragomir Radev and
               Yejin Choi and
               Noah A. Smith},
  title     = {Twist Decoding: Diverse Generators Guide Each Other},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.09273},
}
```
<p align="center">
<a href="https://www.cs.washington.edu/research/nlp">
<img src="https://github.com/jungokasai/THumB/blob/master/figs/uwnlp_logo.png" height="100" alt="UWNLP Logo">
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://allenai.org/">
<img src="https://github.com/jungokasai/THumB/blob/master/figs/ai2_logo.png" height="100" alt="AI2 Logo" style="padding-right:160">
</a>
</p>
