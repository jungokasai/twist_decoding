import torch, argparse, os
from os.path import join
from fairseq.models.bart import BARTModelRerank, BARTModelTwist
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
                   split
                   ):
    checkpoint_dirs = checkpoint_dirs.split(':')
    checkpoint_files = checkpoint_files.split(':')
    data_dirs = data_dirs.split(':')
    model_names = [checkpoint_file.split("_")[1].replace(".tldr", "").replace(".pt", "") for checkpoint_file in checkpoint_files]
    bart_models = [
        BARTModelTwist.from_pretrained(
        checkpoint_dirs[0],
        checkpoint_file=checkpoint_files[0],
        data_name_or_path=data_dirs[0] + '-bin',
        task='translation',
        ),
        BARTModelRerank.from_pretrained(
        checkpoint_dirs[1],
        checkpoint_file=checkpoint_files[1],
        data_name_or_path=data_dirs[1] + '-bin',
        task='translation',
        ),
    ]
    data_models = []
    for data_dir in data_dirs:
        data, nb_sents, nb_batches = read_input(join(data_dir, '{}.source'.format(split)), batch_size)
        data_models.append(data)
    assert len(checkpoint_dirs) == 2
    assert len(checkpoint_files) == 2
    assert len(data_dirs) == 2
    assert len(data_models) == 2
    for bart_model in bart_models:
        if torch.cuda.is_available():
                bart_model.cuda()
                bart_model.half()
        bart_model.eval()
    outputs = [[] for _ in range(len(bart_models))]

    for batch_idx in range(nb_batches):
        print('Batch ID: {}/{}'.format(batch_idx, nb_batches))
        src_batch = [data[batch_idx*batch_size:(batch_idx+1)*batch_size] for data in data_models]
        candidates = None
        patience_factor = 1.0
        for model_idx in range(len(bart_models)):
            with torch.no_grad():
                hypotheses_batch = bart_models[model_idx].sample(
                                                src_batch[model_idx],
                                                beam=beam, 
                                                lenpen=lenpen, 
                                                max_len_b=max_len_b,
                                                min_len=min_len,
                                                no_repeat_ngram_size=no_repeat_ngram_size,
                                                patience_factor=patience_factor,
                                                candidates = candidates,
                                                )
                candidates, scores_batch, encoder_outs_batch = hypotheses_batch[0], hypotheses_batch[1], hypotheses_batch[1]
                candidates = pad_candidates(candidates, int(beam*patience_factor))
                hypotheses_batch = [hypos[0] for hypos in candidates]
                outputs[model_idx].extend(hypotheses_batch)

    with open(out_file, 'wt') as fout:
        for output in outputs[-1]:
            fout.write(output)
            fout.write('\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out-file', type=str, metavar='N',
                        default='summ/scitldr/output/test.rerank.txt', help='target output file')
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
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    args = parser.parse_args()

    generate_TLDRs(**vars(args))
