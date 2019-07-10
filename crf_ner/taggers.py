import nltk


class NLTK_AVGP_Tagger(object):
    def __init__(self):
        self._tagger = None

    def __call__(self, tokens):
        if self._tagger is None:
            self._tagger = nltk.PerceptronTagger()
        return [tag for _, tag in self._tagger.tag(tokens)]


TAGGERS = {
    'en': NLTK_AVGP_Tagger(),
}