import os
import datasets
import argparse
import tiktoken
import numpy as np

from tqdm import tqdm
from openai import OpenAI

from src.utils.util import set_seeds
from src.data.utils import get_dataset

def download_and_process_dataset(dataset_name):

    dataset = datasets.load_dataset(dataset_name)
    if 'validation' not in dataset:
        tmp_dataset = dataset['train'].train_test_split(test_size=0.2, seed=0)
        dataset['train'] = tmp_dataset['train']
        dataset['validation'] = tmp_dataset['test']
    dataset.save_to_disk(os.path.join('data', dataset_name))

    # Load examples from the ag_news dataset
    dataset, _, _, text_column_name = get_dataset('data/'+dataset_name)

    # take only the validation split
    validation_dataset = dataset['validation']

    # tokenize the examples in the validation split
    # using the tiktoken tokenizer for gpt-3.5-turbo
    enc = tiktoken.get_encoding('cl100k_base')
    tokenized_examples = list(map(enc.encode, validation_dataset[text_column_name]))
    total_tokens = sum([len(example) for example in tokenized_examples])
    print(f'Total number of examples: {len(tokenized_examples)}')

    # Get the embeddings using OpenAI's API
    client = OpenAI()

    texts = validation_dataset[text_column_name]
    embeddings = []
    for idx in tqdm(range(len(texts))):
        output = client.embeddings.create(input = texts[idx:idx+1], 
                                          model='text-embedding-3-small')
        embedding = output.data[0].embedding
        embeddings.append(embedding)
    
    # Save these embeddings in the dataset directory
    embeddings = np.array(embeddings)
    np.save(os.path.join(f'data/{dataset_name}', f'validation_openai_v3_embeddings.npy'), embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True)
    args = parser.parse_args()

    set_seeds(0)
    download_and_process_dataset(args.dataset)
