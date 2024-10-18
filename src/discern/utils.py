import os
import torch
import pickle
import numpy as np

from functools import partial
from typing import List, Callable
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.utils.util import device
from src.utils.Config import Config
from src.data.utils import get_dataset
from src.discern.tagging import evaluate_cluster_description
from src.discern.cluster import ClusterSet

def get_predicate_eval_fn(pretrained_weight: str):

    evaluator_tokenizer = AutoTokenizer.from_pretrained(pretrained_weight)

    if 'Mixtral-8x7B-Instruct-v0.1' in pretrained_weight:
        # Load this model in 4-bit for tractable inference
        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
        evaluator_model = AutoModelForCausalLM.from_pretrained(
                            pretrained_weight,
                            device_map='sequential',
                            quantization_config=bnb_config,
                        )
    else:
        evaluator_model = AutoModelForCausalLM.from_pretrained(
                            pretrained_weight,
                            torch_dtype=torch.bfloat16,
                            device_map='sequential',
                        ).to(device)
    eval_description = partial(evaluate_cluster_description, tokenizer=evaluator_tokenizer, model=evaluator_model)
    return eval_description

def compute_score(sentences: List[str], 
              description: str,
              other: bool=False, 
              eval_description: Callable=None
    ):
    '''
    Get the classification score for the sentences and the description.
    Input:
    - sentences: list of sentences
    - description: description to evaluate
    - other: boolean to evaluate the description without any sentences
    - eval_description: function to evaluate the description
    Output:
    - identified: list of sentences identified by the description
    - percentage_identified: percentage of sentences identified by the description
    '''

    classifications = eval_description(sentences, description)
    identified = [sentence for sentence, identified in zip(sentences, classifications) if identified == 'yes']

    if other:
        null_classifications = eval_description(sentences, "")
        null_identified = [sentence for sentence, identified in zip(sentences, null_classifications) if identified == 'yes']

        # Their difference represents the instances really identified
        others_not_null_identified = list(set(identified) - set(null_identified))
        return identified, len(others_not_null_identified) / len(sentences)
    else:
        return identified, len(identified) / len(sentences)

def get_clusterset(config: Config,
                   validation_dataset: Dataset,
                   labels: np.array,
                   logits: np.array,
                   label_names: List[str],
                   input_representations: np.array) -> ClusterSet:
    '''
    Get the clusterset for the data.
    Input:
    - config: configuration
    - validation_dataset: Dataset to perform clustering over
    - labels: labels
    - logits: logits
    - input_representations: representations

    Output:
    - clusterset: clusterset
    '''

    try:
        clusterset_name = f'ClusterSet_{config.cluster_embeddings}_{config.clustering_mode}_P{int(config.is_pca)}_L{int(config.include_logits)}_C{int(config.cluster_by_class)}'
        if config.distance_threshold is not None:
            clusterset_name += f'_DT{config.distance_threshold}'
        cluster_path = os.path.join(config.classifier, f'{clusterset_name}.pkl')
        assert os.path.exists(cluster_path), f'ClusterSet {clusterset_name} does not exist!'
        config.logger.info(f'Loading ClusterSet {clusterset_name}...')
        clusterset = pickle.load(open(cluster_path,'rb'))
        
    except:
        config.logger.info('ClusterSet not found. Creating new clusterset.')
        clusterset = ClusterSet(config, validation_dataset, labels, logits, label_names,
                            input_representations, distance_threshold=config.distance_threshold)
        clusterset.cluster()
        clusterset.print_misclassification_rates()
        pickle.dump(clusterset, open(os.path.join(config.classifier, f'{clusterset}.pkl'),'wb'))
    
    return clusterset
        

def load_cluster_inputs(config: Config):
    '''
    Load the predictions and representations for the clustering.
    Input:
    - config: configuration
    Output:
    - dataset: dataset
    - labels: labels
    - logits: logits
    - input_representations: input representations
    - base_misclassify_rate: base misclassification rate
    '''

    # Load the predictions and representations on the evaluation set
    with open(f'{config.classifier}/eval_results.npz', 'rb') as f:
        eval_results = np.load(f)
        logits, labels = eval_results['logits'], eval_results['labels']
    # print the classifier accuracy
    config.logger.info(f'Number of examples in the validation set: {len(labels)}')
    config.logger.info(f'Classifier accuracy: {np.mean(logits.argmax(axis=1) == labels) * 100 :.2f}%')
    config.logger.info(f'Base misclassification rate: {np.mean(logits.argmax(axis=1) != labels) * 100 :.2f}%')
    base_misclassify_rate = np.mean(logits.argmax(axis=1) != labels)

    dataset, _, label2id, text_column_name = get_dataset(config.dataset)

    # Sample num_train_examples points from the training set
    train_idxs = np.load(f'{config.classifier}/train_idxs.npz')['idxs']
    dataset["train"] = dataset["train"].select(train_idxs)
    # validation set will be equal or less than training set size
    valid_idxs = np.load(f'{config.classifier}/valid_idxs.npz')['idxs']
    dataset["validation"] = dataset["validation"].select(valid_idxs)

    with open(f'{config.dataset}/validation_openai_v3_embeddings.npy', 'rb') as f:
        input_representations = np.load(f)[valid_idxs]

    examples = dataset['validation'][text_column_name]
    label_names = list(label2id.keys())
    return examples, labels, logits, label_names, input_representations, base_misclassify_rate