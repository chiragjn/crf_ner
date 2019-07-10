import logging
from gensim.models import KeyedVectors
from .common import ASSETS_DIR

class GLOVE25_Embedder(object):
    def __init__(self):
        self._word_vectors = None
        self._oov_term = None

    def __call__(self, tokens):
        if self._word_vectors is None:
            logging.info('Loading glove vector, this will take some time ...')
            self._word_vectors = KeyedVectors.load_word2vec_format(str(ASSETS_DIR / 'kv.glove.twitter.27B.25d.txt'))
            self._oov_term = self._word_vectors.index2entity[-1]
            logging.info('Glove load done')
        
        _tokens = [token if token in self._word_vectors else self._oov_term for token in tokens]
        vectors = self._word_vectors[_tokens]
        return vectors


EMBEDDERS = {
    'en': GLOVE25_Embedder(),
}
