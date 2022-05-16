import argparse, os
from fairseq.models.transformer import TransformerModelBaseTwist
from twist.utils import get_line_ids, pad_candidates
import torch
        
def translate(model_dirs,
              in_file,
              out_file,
              batch_size,
              model_names,
              num_shards,
              shard_id,
              lenpen,
              beam,
              src_lang,
              tgt_lang,
              r2l,
              lmd_g,
              lmd_f,
              max_updates,
              ):
    model_dirs = model_dirs.split(':')
    model_names = model_names.split(':')
    r2l = [bool(float(x)) for x in r2l.split(':')]
    assert len(model_dirs) == len(model_names)
    assert len(model_dirs) == len(r2l)
    # It's reranking. So always two models or fewer!
    assert len(model_dirs) <= 2
    models = [] 
    src_bpe, tgt_bpe = 'codes.' + src_lang, 'codes.' + tgt_lang
    for model_dir, model_name in zip(model_dirs, model_names):
        model = TransformerModelBaseTwist.from_pretrained(model_dir,
                                                 checkpoint_file=model_name,
                                                 data_name_or_path=model_dir,
                                                 bpe='fastbpe',
                                                 bpe_codes=os.path.join(model_dir, src_bpe),
                                                 tgt_bpe='fastbpe',
                                                 tgt_bpe_codes=os.path.join(model_dir, tgt_bpe),
                                                 tokenizer=None,
                                                 tgt_tokenizer=None,
                                                 source_lang=src_lang,
                                                 target_lang=tgt_lang,
                                                 )
        model.cuda()
        model.eval()
        models.append(model)

    start_id, end_id = get_line_ids(in_file, num_shards, shard_id)
    print(start_id, end_id)
    src_sents = []
    with open(in_file) as fin:
        for i, line in enumerate(fin):
            if start_id <= i < end_id:
                line = line.strip()
                src_sents.append(line)
    nb_sents = len(src_sents)
    nb_batches = (nb_sents+batch_size-1)//batch_size
    outputs = [[] for _ in range(max_updates)]
    for batch_idx in range(nb_batches):
        print('Batch ID: {}/{}'.format(batch_idx, nb_batches))
        candidates = None
        prev_candidates = None
        src_text = src_sents[batch_idx*batch_size:(batch_idx+1)*batch_size] 
        output_idx = 0
        encoder_outs_models = [None for _ in range(len(models))]
        for step in range(max_updates//len(models)+1):
            for model_idx, model, r2l_model in zip(range(len(models)), models, r2l):
                if candidates is None:
                    patience_factor = 1
                else:
                    # always put 2 for better search with penalty terms
                    patience_factor = 2
                if model_idx == 0:
                    lmd = lmd_g
                else:
                    assert model_idx == 1
                    lmd = lmd_f
                encoder_outs = encoder_outs_models[model_idx]
                candidates, batched_hypos, encoder_outs = model.translate(src_text, lenpen=lenpen, beam=beam, candidates=candidates, r2l=r2l_model, lmd=lmd, patience_factor=patience_factor, encoder_outs=encoder_outs)
                # Only need to compute encoder_outs once per model.
                if step == 0:
                    encoder_outs_models[model_idx] = encoder_outs
                # In a rare edge case, we get fewer than beam*patience_factor hypotheses.
                # In this case, repeat the first sequence to pad.
                # If we get an empty set, use previous candidates.
                candidates = pad_candidates(candidates, beam*int(patience_factor), prev_candidates)
                prev_candidates = candidates
                output = [hypos[0] for hypos in candidates]
                outputs[output_idx].extend(output)
                if output_idx == (max_updates - 1):
                    break
                output_idx += 1
    for output_idx in range(max_updates):
        with open(out_file + '_update-{}.out'.format(str(output_idx)), 'wt') as fout:
            for output in outputs[output_idx]:
                fout.write(output)
                fout.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--batch-size', default=20, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--model-dirs', type=str, metavar='N', 
                        help='model directories')
    parser.add_argument('--model-names', type=str, metavar='N', 
                        help='model name')
    parser.add_argument('--r2l', type=str, metavar='N', 
                        default='0:0', help='Are they r2l?')
    parser.add_argument('--in-file', type=str, metavar='N',
                        help='source file')
    parser.add_argument('--out-file', type=str, metavar='N',
                        default='mt/wmt/zh-en/output/test.twist', help='target output file')
    parser.add_argument('--num-shards', default=1, type=int, metavar='N',
                        help='number of shards')
    parser.add_argument('--shard-id', default=0, type=int, metavar='N',
                        help='shard id')
    parser.add_argument('--lenpen', default=1.0, type=float, metavar='N',
                        help='length penalty')
    parser.add_argument('--beam', default=5, type=int, metavar='N',
                        help='beam size')
    parser.add_argument('--src-lang', type=str, metavar='N', 
                        default='de', help='source language')
    parser.add_argument('--tgt-lang', type=str, metavar='N', 
                        default='en', help='target language')
    parser.add_argument('--max-updates', default=1, type=int, metavar='N',
                        help='max updates')
    parser.add_argument('--lmd-g', default=1.0, type=float,
                        help='lambda g')
    parser.add_argument('--lmd-f', default=1.0, type=float,
                        help='lambda f')
    args = parser.parse_args()
    with torch.no_grad():
        translate(args.model_dirs,
                  args.in_file,
                  args.out_file,
                  args.batch_size,
                  args.model_names,
                  args.num_shards,
                  args.shard_id,
                  args.lenpen,
                  args.beam,
                  args.src_lang,
                  args.tgt_lang,
                  args.r2l,
                  args.lmd_g,
                  args.lmd_f,
                  args.max_updates,
                  )
