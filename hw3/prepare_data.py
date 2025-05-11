

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'


import pandas as pd
from tqdm.auto import tqdm
from torchtext.data import Field, Example, Dataset, BucketIterator


word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
fields = [('source', word_field), ('target', word_field)]

data = pd.read_csv('news.csv', delimiter=',')


examples = []
for _, row in tqdm(data.iterrows(), total=len(data)):
    source_text = word_field.preprocess(row.text)
    target_text = word_field.preprocess(row.title)
    examples.append(Example.fromlist([source_text, target_text], fields))

def import_dataset():
    dataset = Dataset(examples, fields)

    train_dataset, test_dataset = dataset.split(split_ratio=0.85)

    word_field.build_vocab(train_dataset, min_freq=7)

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
)