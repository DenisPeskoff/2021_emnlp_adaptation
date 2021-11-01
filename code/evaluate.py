import re
import logging
import argparse
import rank_metrics

logger = logging.getLogger('modulation.evaluate')
VERBOSE_REMOVE_PATTERN = re.compile(':.+$')


def calculate_acc(predictions, gold, topn=1, oov_token='_OOV_',
                  lowercase=False):
    assert len(predictions) == len(gold)
    assert len(predictions) > 0

    top = 0.0
    n = 0
    n_oov = 0
    lc = (lambda x: x.lower()) if lowercase else (lambda x: x)

    for p, g in zip(predictions, gold):
        p = set(p[:topn])
        g = set(g)

        assert oov_token not in g, 'Gold contains the token reserved for OOVs'

        n += 1
        if len(p) == 1 and oov_token in p:
            n_oov += 1
            continue

        p = set(map(lc, p))
        g = set(map(lc, g))

        if len(p & g) > 0:
            top += 1

    return top/n, top/(n-n_oov), n, n_oov, top


def load_similarity_file(file, sort=False, sim_th=None, topn=-1):
    """
    :return dict -> list
    """
    res = dict()

    with open(file, 'r') as fin:
        for line in fin:
            data = line.split('\t')
            word = data[0]

            if word in res:
                logger.warning('Word {} was already seen. Ignoring!'.format(word))
                continue

            res[word] = [(data[i*2 + 1], float(data[i*2 + 2])) for i in range(int((len(data)-1)/2)) if sim_th is None or float(data[i*2 + 2]) > sim_th]
            if sort:
                res[word] = sorted(res[word], key=lambda x: x[1], reverse=True)

            if topn > 0:
                res[word] = res[word][:topn]

    return res


def replace_space(word, char):
    if char is not None:
        word = word.replace(' ', char)
    return word


def read_gold(inps, rep_space=None):
    res = dict()

    for inp in inps:
        with open(inp) as fin:
            for line in fin:
                line = line.strip().split('\t')
                src = replace_space(line[0], rep_space)

                for w in line[1:]:
                    res.setdefault(src, set()).add(replace_space(w, rep_space))

    return res


def get_prediction_vectors(inp, gold):
    pred_sims = load_similarity_file(inp, sort=True)

    res = list()

    for src, preds in pred_sims.items():
        src = VERBOSE_REMOVE_PATTERN.sub('', src)

        if src not in gold:
            logger.warning(f'No gold annotation, ignoring word: {src}')
            continue

        res.append([int(w in gold[src]) for w, v in preds])
        if sum(res[-1]) > 0:
            logger.info(f'Correct modulation for: {src}')

    return res


def get_pred_gold_for_acc(inp, gold):
    pred_sims = load_similarity_file(inp, sort=True)

    gold_out = list()
    pred_out = list()

    for src, preds in pred_sims.items():
        src = VERBOSE_REMOVE_PATTERN.sub('', src)

        if src not in gold:
            gold_out.append('_PLACEHOLDER_')
            pred_out.append('_OOV_')
        else:
            gold_out.append(gold[src])
            pred_out.append([w for w, v in preds])

    return pred_out, gold_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golds', required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    golds = [args.golds]
    predictions = args.predictions

    logger.info('Loading data...')
    gold = read_gold(golds, '_')  # {src: {trgs}}
    pred = get_prediction_vectors(predictions, gold)

    MRR = rank_metrics.mean_reciprocal_rank(pred)
    MAP = rank_metrics.mean_average_precision(pred)

    print(f'MRR:\t{MRR}\tMAP\t{MAP}')

    logger.warning('OOV means if we have no annotation for a given word!')
    acc_pred, acc_gold = get_pred_gold_for_acc(predictions, gold)
    for i in [1, 5, 10, 50, 100]:
        acc, acc_oov, num, num_oov, _ = calculate_acc(acc_pred, acc_gold,
                                                      topn=i,
                                                      oov_token='_OOV_',
                                                      lowercase=False
                                                      )
        print(f'# all test cases\t{num}\t# OOV\t{num_oov}\tACC@{i}\t{acc}\tACC@{i} OOV\t{acc_oov}')
