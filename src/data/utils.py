from typing import Tuple, Union, Dict
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

def get_tokenized_dataset(dataset: Union[DatasetDict, Dataset], 
                          tokenizer: PreTrainedTokenizerBase,
                          text_column_name: str
                          ) -> Tuple[Union[Dataset, DatasetDict], DataCollatorWithPadding]:
    '''
    Tokenizes the dataset using the tokenizer and returns the tokenized dataset and the data collator.

    Args:
    dataset: Union[DatasetDict, Dataset]: The dataset to be tokenized.
    tokenizer: PreTrainedTokenizerBase: The tokenizer to be used for tokenization.
    text_column_name: str: The name of the column containing the text data.

    Returns:
    Tuple[Union[Dataset, DatasetDict], DataCollatorWithPadding]: The tokenized dataset and the data collator.
    '''
    
    def preprocess_function(examples):
        return tokenizer(examples[text_column_name], truncation=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_dataset, data_collator

def convert_label(x: Dict, 
                  label2id: Dict):
    '''
    converts the label to its corresponding id.
    '''
    return label2id[x['label']]

def get_dataset(
        dataset_name: str ='data/ag_news'
    )-> Tuple[Union[Dataset, DatasetDict], dict, dict, str]:

    '''
    Returns the dataset, id2label, label2id and text_column_name for the given dataset_name.

    Args:
    dataset_name: str: The name of the dataset.

    Returns:
    dataset: The dataset.
    id2label: The id to label mapping.
    label2id: The label to id mapping.
    text_column_name: The name of the column containing the text data.
    '''

    if dataset_name == 'data/covid':
        dataset = load_from_disk('data/covid')
        id2label = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Extremely Positive", 4: "Extremely Negative"}
        label2id = {v:k for k,v in id2label.items()}
        dataset = dataset.map(lambda x: {'label': convert_label(x, label2id)})
        text_column_name = "text"
    elif dataset_name == 'data/trec':
        dataset = load_from_disk('data/trec')
        id2label = {0: "abbreviation", 1: "entity", 2: "description", 3: "human", 4: "location", 5: "number"}
        label2id = {v:k for k,v in id2label.items()}
        text_column_name = "text"
    elif dataset_name == 'data/ag_news':
        dataset = load_from_disk('data/ag_news')
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label2id = {v:k for k,v in id2label.items()}
        text_column_name = "text"
    
    return dataset, id2label, label2id, text_column_name