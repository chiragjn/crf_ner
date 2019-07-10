import nltk

class NLTK_Tokenizer(object):
    def __init__(self):
        self._tokenizer = None
        self._sent_tokenizer = None

    def __call__(self, text):
        if self._sent_tokenizer is None:
            self._tokenizer = nltk.load('tokenizers/punkt/{0}.pickle'.format('english'))
            self._sent_tokenizer = self._tokenizer.tokenize
        sentences = self._sent_tokenizer(text)
        tokens = []
        for sent in sentences:
            tokens.extend(nltk.word_tokenize(sent, preserve_line=True))
        return tokens

TOKENIZERS = {
    'en': NLTK_Tokenizer(),
}