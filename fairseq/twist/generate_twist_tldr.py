import torch, argparse, os
from os.path import join
from fairseq.models.bart import BARTModelTwist
from twist.utils import get_line_ids, pad_candidates, read_input

def generate_TLDRs(
                   out_file,
                   batch_size,
                   checkpoint_dirs,
                   checkpoint_files,
                   data_dirs,
                   beam,
                   lenpen,
                   max_len_b,
                   min_len,
                   no_repeat_ngram_size,
                   lmd_g,
                   lmd_f,
                   max_updates,
                   split
                   ):
    checkpoint_dirs = checkpoint_dirs.split(':')
    checkpoint_files = checkpoint_files.split(':')
    data_dirs = data_dirs.split(':')
    model_names = [checkpoint_file.split("_")[1].replace(".tldr", "").replace(".pt", "") for checkpoint_file in checkpoint_files]
    bart_models = [BARTModelTwist.from_pretrained(
        checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_dir + '-bin',
        task='translation',
    ) for checkpoint_dir, checkpoint_file, data_dir in zip(checkpoint_dirs, checkpoint_files, data_dirs)
    ]
    data_models = []
    for data_dir in data_dirs:
        data, nb_sents, nb_batches = read_input(join(data_dir, '{}.source'.format(split)), batch_size)
        data_models.append(data)
    lmds = [lmd_g, lmd_f]
    for bart_model in bart_models:
        if torch.cuda.is_available():
                bart_model.cuda()
                bart_model.half()
        bart_model.eval()
    outputs = [[] for _ in range(max_updates)]

    for batch_idx in range(nb_batches):
        print('Batch ID: {}/{}'.format(batch_idx, nb_batches))
        src_batch = [data[batch_idx*batch_size:(batch_idx+1)*batch_size] for data in data_models]
        candidates = None
        prev_candidates = None
        patience_factor = 1.0
        encoder_outs_models = [None for _ in range(len(bart_models))]
        output_idx = 0
        for step in range(max_updates//len(bart_models)+1):
            for model_idx in range(len(bart_models)):
                with torch.no_grad():
                    hypotheses_batch = bart_models[model_idx].sample(
                                                    src_batch[model_idx],
                                                    beam=beam, 
                                                    lenpen=lenpen, 
                                                    max_len_b=max_len_b,
                                                    min_len=min_len,
                                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                                    lmd=lmds[model_idx],
                                                    patience_factor=patience_factor,
                                                    candidates = candidates,
                                                    encoder_outs = encoder_outs_models[model_idx],
                                                    )
                    candidates, scores_batch, encoder_outs_batch = hypotheses_batch[0], hypotheses_batch[1], hypotheses_batch[1]
                    if step == 0:
                        encoder_outs_models[model_idx] = encoder_outs_batch
                    candidates = pad_candidates(candidates, int(beam*patience_factor), prev_candidates)
                    prev_candidates = candidates
                    hypotheses_batch = [hypos[0] for hypos in candidates]
                    outputs[output_idx].extend(hypotheses_batch)
                    if output_idx == (max_updates - 1):
                        break
                    patience_factor = 2.0
                    output_idx += 1

    for output_idx in range(max_updates):
        with open(out_file + '_update-{}.txt'.format(str(output_idx)), 'wt') as fout:
            for output in outputs[output_idx]:
                fout.write(output)
                fout.write('\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out-file', type=str, metavar='N',
                        default='summ/scitldr/output/test.twist', help='target output file')
    parser.add_argument('--checkpoint-dirs', help='Path to checkpoint directory')
    parser.add_argument('--data-dirs', help='Path to data directory')
    parser.add_argument('--checkpoint-files', default='checkpoint_best.pt')
    
    # Decoder params
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--beam', default=5, type=int)
    parser.add_argument('--lenpen', default=1.0, type=float)
    parser.add_argument('--max_len_b', default=30, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    parser.add_argument('--lmd-g', default=1.0, type=float,
                        help='initial lamda')
    parser.add_argument('--lmd-f', default=1.0, type=float,
                        help='initial lamda')
    parser.add_argument('--max-updates', default=5, type=int)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    args = parser.parse_args()

    generate_TLDRs(**vars(args))
