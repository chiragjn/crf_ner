from crf_ner.model import CRFModel, DEFAULT_FEATURES
from crf_ner.utils import read_tsv

def main():
    delimiter = '\t'
    data_format = 'tsv'
    features = DEFAULT_FEATURES
    train_data_path = 'example_exp/data.train.txt'
    val_data_path = 'example_exp/data.val.txt'
    save_path = 'example_exp/models/entity_model'
    train_data = read_tsv(train_data_path, delimiter=delimiter)
    val_data = read_tsv(val_data_path, delimiter=delimiter)
    model = CRFModel(features, language='en')
    model.train(train_data, max_iter=1000)
    model.save(save_path)
    model = CRFModel.load(save_path)
    print(model.predict(val_data))

if __name__ == '__main__':
    main()
