import os
import wget
import zipfile

import nltk
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path

THIS_DIR = Path(__file__).resolve()
ASSETS_DIR = THIS_DIR / 'assets'
MODELS_DIR = THIS_DIR / 'models'

def setup():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def download_nltk_assets():
    assets = [
        'punkt',
        'maxent_treebank_pos_tagger',
        'averaged_perceptron_tagger',
    ]

    for asset in assets:
        print("Downloading {} ...".format(asset), end=" ")
        status = nltk.download(asset)
        print("OK" if status else "Failed")
    
def download_glove25d():
    outfile = ASSETS_DIR / 'kv.glove.twitter.27B.25d.txt'
    outfile = str(outfile)
    if not os.path.exists(outfile):
        wget.download(url='http://nlp.stanford.edu/data/glove.twitter.27B.zip')
        glove_zip = zipfile.ZipFile('glove.twitter.27B.zip', 'r')
        glove_zip.extract(member='glove.twitter.27B.25d.txt')
        os.remove('glove.twitter.27B.zip')
        glove2word2vec('glove.twitter.27B.25d.txt', outfile)


if __name__ == '__main__':
    setup()
    download_nltk_assets()
    download_glove25d()
