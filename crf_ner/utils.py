import logging

def findall(text, substring):
    offset = 0
    offset = text.find(substring, offset)
    while offset != -1:
        yield offset, offset + len(substring)
        offset += len(substring) 
        offset = text.find(substring, offset)

def make_json_instance_from_line(line, delimiter="|||"):
    text, *entities = line.split(delimiter)
    logging.info("====================")
    logging.info(text)
    instance = {"text": text, "entities": []}
    for entity in entities:
        fnd, num, label = entity.split(':')
        spans = list(findall(text, fnd))
        if num:
            num_i = int(num)
            spans = [spans[num_i]]
        for start, end in spans:
            entity = {"start": start, "end": end, "label": label} 
            instance["entities"].append(entity)
            logging.info('%s > %s', text[entity["start"]:entity["end"]], label)
    return instance

def read_tsv(filepath, delimiter='\t'):
    dataset = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            instance = make_json_instance_from_line(line, delimiter=delimiter)
            dataset.append(instance)
    return dataset