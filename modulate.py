import numpy as np
import logging
from tqdm import tqdm
import argparse
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('modulate')


class Procrustes():
    def __init__(self):
        self._w = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        u, s, vt = np.linalg.svd(y.T.dot(x))
        self._w = vt.T.dot(u.T)

    def predict(self, x):
        if self._w is None:
            raise "First fit!"

        x = np.array(x)

        return x.dot(self._w)


def save_similarities(sims, output, topn=-1, float_precision=4):
    with open(output, 'w') as fout:
        for k, v in sims.items():
            tmp = ['{}\t{}'.format(w, format(s, '.{}f'.format(float_precision))) for w, s in v]
            if topn > 0:
                tmp = tmp[:topn]
            fout.write('{}\t{}\n'.format(k, '\t'.join(tmp)))


def load_train(inp, topn=-1):
    res = list()
    with open(inp) as fin:
        for line in fin:
            line = line.strip().split('\t')

            tmp = line[1:1+topn] if topn > 0 else line[1:]
            for item in tmp:
                res.append((line[0], item))

    return res


def train_projection(train, src_emb, trg_emb, model):
    """
    train: list of pairs of words: [(src, trg),]
    """
    x, y = [], []

    for sw, tw in train:
        if sw in src_emb and tw in trg_emb:
            # FIXME use lst_emb based averaging here as well
            x.append(src_emb[sw])
            y.append(trg_emb[tw])

    assert len(x)  # implicit check of y as well
    model.fit(x, y)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Words to modulate")
    parser.add_argument('--output', required=True,)
    parser.add_argument('--topn', default=100, type=int, help="Top N modulations to save")
    parser.add_argument('--src_emb', required=True, help="Embeddings of the source language")
    parser.add_argument('--trg_emb', required=True, help="Embeddings of the target language")
    parser.add_argument('--vocab_limit', default=None, type=int, help="Limit the vocabulary of the embeddings")
    parser.add_argument('--src_pos', default=['USA'], nargs="+",
                        help="This parameter is used for the unsupervised"
                        " methods. Sets the target culture of the adaptation."
                        " The given token is looked up in src_emb. src_pos and"
                        " trg_pos are averaged. Multiple values can be given."
                        )
    parser.add_argument('--trg_pos', default=['United_States'], nargs="+",
                        help="This parameter is used for the unsupervised"
                        " methods. Sets the target culture of the adaptation."
                        " The given token is looked up in trg_emb. src_pos and"
                        " trg_pos are averaged. Multiple values can be given."
                        )
    parser.add_argument('--src_neg', default=['Deutschland'], nargs="+",
                        help="This parameter is used for the unsupervised"
                        " methods. Sets the source culture of the adaptation."
                        " The given token is looked up in src_emb. src_neg and"
                        " trg_neg are averaged. Multiple values can be given."
                        )
    parser.add_argument('--trg_neg', default=['Germany'], nargs="+",
                        help="This parameter is used for the unsupervised"
                        " methods. Sets the source culture of the adaptation."
                        " The given token is looked up in trg_emb. src_neg and"
                        " trg_neg are averaged. Multiple values can be given."
                        )
    parser.add_argument('--method', default='add', help="Unsupervised methods: add, mul, cos. Supervised: ridge, orthogonal")
    parser.add_argument('--train_file', default=None, help="Training data if a supervised method is used.")
    parser.add_argument('--verbose', default=1, type=int, help="Extra logging.")
    args = parser.parse_args()

    words = args.input
    output = args.output
    src_emb = args.src_emb
    trg_emb = args.trg_emb
    topn = args.topn
    vocab_limit = args.vocab_limit
    src_pos = args.src_pos
    trg_pos = args.trg_pos
    src_neg = args.src_neg
    trg_neg = args.trg_neg
    method = args.method
    train_file = args.train_file
    verbose = args.verbose != 0

    logger.info('Loading embedding files')
    src_emb = KeyedVectors.load_word2vec_format(
        src_emb, binary=False, limit=vocab_limit)
    trg_emb = KeyedVectors.load_word2vec_format(
        trg_emb, binary=False, limit=vocab_limit)

    if train_file is None:
        assert all([w in src_emb for w in src_neg+src_pos])
        assert all([w in trg_emb for w in trg_neg+trg_pos])

    # lst_emb = [src_emb, trg_emb]
    lst_emb = [src_emb]

    model = None
    if train_file is not None:
        logger.info('Loading training pairs...')
        train = load_train(train_file, -1)

        logger.info('Training model...')
        if method == 'ridge':
            model = Ridge(alpha=0.0, random_state=0)
        elif method == 'orthogonal':
            model = Procrustes()
        else:
            raise f"Not supported supervised model: {method}"

        model = train_projection(train, src_emb, trg_emb, model)

    logger.info('Modulating...')
    res = dict()
    with open(words, 'r') as fin:
        for word in tqdm(fin):
            word = word.strip()
            word_orig = word

            if all([word not in emb for emb in lst_emb]):
                if '_' in word:
                    for w in word.split('_')[::-1]:
                        if any([w in emb for emb in lst_emb]):
                            logger.info(f'Backing-off: {word} -> {w}')
                            word = w
                            if verbose > 0:
                                word_orig = f'{word_orig}:{word}'
                            break
                    else:
                        logger.info('OOV: {}'.format(word_orig))
                        res[word_orig] = [('OOV', 0.0)]
                        continue
                else:
                    logger.info('OOV: {}'.format(word_orig))
                    res[word_orig] = [('OOV', 0.0)]
                    continue

            if model is None:
                if method == 'cos':
                    res[word_orig] = trg_emb.most_similar(
                        positive=[
                            np.mean(
                                [
                                    emb[word]
                                    for emb in lst_emb if word in emb
                                ],
                                axis=0
                            )
                        ],
                        topn=topn
                    )
                else:
                    most_sim = trg_emb.most_similar if method == 'add' else trg_emb.most_similar_cosmul

                    res[word_orig] = most_sim(
                        positive=[
                            np.mean(
                                [
                                    emb[word]
                                    for emb in lst_emb if word in emb
                                ],
                                axis=0
                            ),
                            np.mean(
                                [src_emb[w] for w in src_pos]
                                + [trg_emb[w] for w in trg_pos],
                                axis=0
                            )
                        ],
                        negative=[np.mean(
                            [src_emb[w] for w in src_neg]
                            + [trg_emb[w] for w in trg_neg],
                            axis=0
                        )],
                        topn=topn
                    )
            else:
                res[word_orig] = trg_emb.most_similar(model.predict(
                    np.mean(
                        [
                            emb[word]
                            for emb in lst_emb if word in emb
                        ],
                        axis=0
                    ).reshape(1, -1)
                ), topn=topn)

    logger.info('Saving results')
    save_similarities(res, output)
