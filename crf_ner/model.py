import os
import json
import itertools
from pathlib import Path
import logging

import nltk
import sklearn_crfsuite
import joblib

from .tokenizers import TOKENIZERS
from .taggers import TAGGERS
from .embedders import EMBEDDERS

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

class Token(object):
    def __init__(self, text, offset, end):
        # type: (str, int, int) -> None
        self.text = text
        self.offset = offset
        self.end = end
        self.pos = None
        self.vector = None


def get_token_offsets(text, tokens):
    # type -> Iterator[Token]
    offset = 0
    for token in tokens:
        offset = text.find(token, offset)
        yield Token(token, offset, offset + len(token))
        offset += len(token)

USE_BILOU = False


FEATURE_FUNCTIONS = {
    "low": lambda token: token.text.lower(),
    "title": lambda token: token.text.istitle(),
    "prefix2": lambda token: token.text[:2],
    "prefix3": lambda token: token.text[:2],
    "prefix5": lambda token: token.text[:5],
    "suffix5": lambda token: token.text[-5:],
    "suffix3": lambda token: token.text[-3:],
    "suffix2": lambda token: token.text[-2:],
    "pos": lambda token: token.pos,
    "pos2": lambda token: token.pos[:2],
    "bias": lambda token: "bias",
    "upper": lambda token: token.text.isupper(),
    "digit": lambda token: token.text.isdigit(),
    "emb": lambda token: token.vector
}

DEFAULT_FEATURES = {
    -2: ["low", "title", "upper", "digit", "emb", "pos", "suffix2", "suffix3", "suffix5"],
    -1: ["low", "title", "upper", "digit", "emb", "pos", "suffix2", "suffix3", "suffix5"],
    0: ["bias", "low", "title", "upper", "digit", "emb", "pos"],
    1: ["low", "title", "upper", "digit", "emb", "pos", "prefix2", "prefix3", "prefix5"],
    2: ["low", "title", "upper", "digit", "emb", "pos", "prefix2", "prefix3", "prefix5"],
}


def emb_feature_dict(vector, prefix=''):
    return {'{}:{}'.format(prefix, i): float(val) for i, val in enumerate(vector)}


class Instance(object):
    def __init__(self, text, language='en', entities=None):
        # type: (str, str, Tuple[int, int, str]) -> None
        if entities is None:
            entities = []
        self.text = text
        self.language = language
        self.tokens = list(get_token_offsets(text, TOKENIZERS[language](text)))
        
        self.y = None
        self.entities = entities
        if self.entities:
            self.set_labels()

        self.X = None

    def set_features(self, features_config):
        # type: (Dict[int, List[str]]) -> None
        all_feat_names = set(itertools.chain.from_iterable(features_config.values()))
        use_pos_features = "pos" in all_feat_names
        use_emb_features = "emb" in all_feat_names

        _tokens = [token.text for token in self.tokens]
        if use_pos_features:
            pos_tags = TAGGERS[self.language](_tokens)
            for token, tag in zip(self.tokens, pos_tags):
                token.pos = tag

        if use_emb_features:
            vectors = EMBEDDERS[self.language](_tokens)
            for i, token in enumerate(self.tokens):
                token.vector = vectors[i, :]

        self.X = self._to_features(features_config)

    def _to_features(self, features_config):
        features = []
        for token_idx in range(len(self.tokens)):
            token_features = {}
            if token_idx == 0:
                token_features["BOS"] = True
            if token_idx == len(self.tokens) - 1:
                token_features["EOS"] = True
            for ctx_idx in features_config:
                if 0 <= (token_idx + ctx_idx) < len(self.tokens):
                    for feature_name in features_config[ctx_idx]:
                        model_feature_name = '{}:{}'.format(ctx_idx, feature_name)
                        feature_val = FEATURE_FUNCTIONS[feature_name](self.tokens[token_idx + ctx_idx])
                        if feature_name == 'emb':
                            token_features.update(
                                emb_feature_dict(feature_val, prefix=model_feature_name)
                            )
                        else:
                            token_features[model_feature_name] = feature_val
            features.append(token_features)
        return features

    def set_labels(self):
        # type: (str) -> None
        text, entities, tokens = self.text, self.entities, self.tokens
        missing = 'O'

        starts = {token.offset: i for i, token in enumerate(tokens)}
        ends = {token.end: i for i, token in enumerate(tokens)}
        bilou = ["-" for _ in tokens]
        # Handle entity cases
        for start_char, end_char, label in entities:
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            # Only interested if the tokenization is correct
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    bilou[start_token] = "U-%s" % label
                else:
                    bilou[start_token] = "B-%s" % label
                    for i in range(start_token + 1, end_token):
                        bilou[i] = "I-%s" % label
                    bilou[end_token] = "L-%s" % label
        # Now distinguish the O cases from ones where we miss the tokenization
        entity_chars = set()
        for start_char, end_char, label in entities:
            for i in range(start_char, end_char):
                entity_chars.add(i)
        for n, token in enumerate(tokens):
            for i in range(token.offset, token.end):
                if i in entity_chars:
                    break
            else:
                bilou[n] = missing

        tags = bilou
        if not USE_BILOU:
            for i, tag in enumerate(tags):
                if tag[0] == 'L':
                    tags[i] = 'I' + tag[1:]
                elif tag[0] == 'U':
                    tags[i] = 'B' + tag[1:]

        if "-" in tags:
            logging.info('Instance with text: {} and entities: {} is not properly tagged')
        self.y = tags

    def to_dict(self):
        return {
            'text': self.text,
            'language': self.language,
            'entities': [{'start': start, 'end': end, 'label': label} for start, end, label in self.entities],
            'tokens': [token.text for token in self.tokens],
            'labels': self.y,
        }


class CRFModel(object):
    def __init__(self, features_config, language='en', **kwargs):
        self._features_config = features_config
        self._language = language
        self._model = None
        self._hyperparams = {}

    def convert(self, instance):
        # instance = {'text': '', 'entities': [{'start':'', 'end':'', label: ''}]}
        def unpack(instance):
            text = instance['text']
            # language = instance.get('language', 'en')
            language = self._language
            entities = [(e['start'], e['end'], e['label']) for e in instance.get('entities', [])]
            return text, language, entities

        text, language, entities = unpack(instance)
        crf_instance = Instance(
            text=text,
            language=language,
            entities=entities
        )
        
        crf_instance.set_features(self._features_config)

        return crf_instance

    def _make_xy(self, instances):
        data = []
        for instance in instances:
            crf_instance = self.convert(instance)
            data.append(crf_instance)
        X = [crf_instance.X for crf_instance in data]
        y = [crf_instance.y for crf_instance in data]
        return X, y
    
    def train(self, instances, validation_instances=None, algorithm="lbfgs", c1=0.1, c2=0.1, max_iter=200, all_possible_transitions=True):
        logging.info('Training ...')
        train_X, train_y = self._make_xy(instances)
        
        self._model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            # coefficient for L1 penalty
            c1=c1,
            # coefficient for L2 penalty
            c2=c2,
            # stop earlier
            max_iterations=max_iter,
            # include transitions that are possible, but not observed
            all_possible_transitions=all_possible_transitions,
        )

        self._model.fit(train_X, train_y)

        logging.info('Validating ...')
        val_X, val_y = [], []
        if validation_instances:
            val_X, val_y = self._make_xy(validation_instances)

    def save(self, model_path):
        logging.info('Dumping model to %s', model_path)
        model_path = Path(model_path).resolve()
        if model_path.is_dir():
            raise ValueError('model_path should not be a directory. Add a model name like path/to/model/name')
        model_name = model_path.name
        model_dir = model_path.parent

        if not model_dir.exists():
            os.makedirs(model_dir)
        
        crf_model_name = '{}.joblib.model'.format(model_name)
        joblib.dump(self._model, str(model_dir / crf_model_name))
        
        state = {
            'language': self._language,
            'features_config': self._features_config,
            'model_name': crf_model_name,
            'hyperparams': self._hyperparams,
        }
        
        model_path = str(model_dir / model_name)
        json.dump(state, open(model_path, 'w'))
        return model_path

    @classmethod
    def load(cls, model_path):
        logging.info('Loading from %s', model_path)
        state = json.load(open(model_path))
        features_config = {int(key): value for key, value in state['features_config'].items()}
        instance = cls(features_config=features_config, language=state['language'])
        instance._model = joblib.load(str(Path(model_path).parent / state['model_name']))
        instance._hyperparams = state['hyperparams']
        return instance

    # def decode():
    #     if len(entities) > idx:
    #         entity_probs = entities[idx]
    #     else:
    #         entity_probs = None
    #     if entity_probs:
    #         label = max(entity_probs, key=lambda key: entity_probs[key])
    #         if self.component_config["BILOU_flag"]:
    #             # if we are using bilou flags, we will combine the prob
    #             # of the B, I, L and U tags for an entity (so if we have a
    #             # score of 60% for `B-address` and 40% and 30%
    #             # for `I-address`, we will return 70%)
    #             return (
    #                 label,
    #                 sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
    #             )
    #         else:
    #             return label, entity_probs[label]
    #     else:
    #         return "", 0.0

    def predict(self, instances):
        X, _ = self._make_xy(instances)
        return self._model.predict(X)