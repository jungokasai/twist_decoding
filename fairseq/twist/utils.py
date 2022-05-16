import argparse, os

def get_line_ids(in_file, num_shards, shard_id):
    nb_lines = sum(1 for i in open(in_file, 'rb'))
    shard_size = nb_lines//num_shards
    remainder = nb_lines - shard_size*num_shards
    start_id = shard_size*shard_id + min([shard_id, remainder])
    end_id = shard_size*(shard_id+1) + min([shard_id+1, remainder])
    return start_id, end_id

def pad_candidates(candidates, nb_candidates, prev_candidates=None):
    # in a rare edge case, we get fewer than beam*patience_factor hypotheses.
    # In this case, repeat the first sequence to pad.
    # if it's an empty list, use the previous set
    for i, hypos in enumerate(candidates):
        if len(hypos) == 0 or hypos[0] == '':
            if prev_candidates is not None:
                hypos = prev_candidates[i]
                candidates[i] = hypos
        if len(hypos) < nb_candidates:
            candidates[i] = [hypos[0]]*int(nb_candidates-len(hypos)) + hypos
        elif len(hypos) > nb_candidates:
            candidates[i] = hypos[:nb_candidates]
    return candidates

def read_input(in_file, batch_size):
    start_id, end_id = get_line_ids(in_file, 1, 0)
    print(start_id, end_id)
    src_sents = []
    with open(in_file) as fin:
        for i, line in enumerate(fin):
            if start_id <= i < end_id:
                line = line.strip()
                src_sents.append(line)
    nb_sents = len(src_sents)
    nb_batches = (nb_sents+batch_size-1)//batch_size
    return src_sents, nb_sents, nb_batches
