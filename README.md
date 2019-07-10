### CRF NER
-----------

Train NER models using sklearn crfsuite


Goal of this repo is to be able to put down some data into a TSV file and train an independent reasonably accurate NER model quickly. This is something I wrote for my own use-cases but I am putting it out there anyways.

If you looking for stable and more popular NER frameworks (in general any sequence tagging) you should look [here](https://github.com/topics/named-entity-recognition?l=python&o=desc&s=stars). Some popular repos are Flair, Snips, Rasa, DeepPavlov, AllenNLP, NeuroNER, NCRF++, Ludwig and many more. 

TODO:

- Make decoding more accurate by injecting constraints
- Write functionality to calculate metrics - accuracy, precision, recall, f1
- Make this repo work with CONLL format
- Test this repo on CONLL 2003 data and report scores
- Integrate Fasstext embeddings to enable experimentation with other languages
- Make it easier to specify embedders/taggers/features in a single config file, instead of relying on language code
- Allow BILOU tagging format

Acknowledgements:

- Rasa - A lot of this code was inspired from Rasa's Entity extractor. It is good as it is, but very coupled with their entire NLU system.
