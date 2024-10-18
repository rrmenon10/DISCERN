import os
import json
import torch
import argparse
import evaluate
import numpy as np

from transformers import Trainer
from collections import Counter
from datasets import concatenate_datasets, DatasetDict

from src.data.utils import get_dataset, get_tokenized_dataset
from src.discern.utils import get_clusterset, load_cluster_inputs, get_predicate_eval_fn
from src.utils.util import ParseKwargs, set_seeds, get_descriptions
from src.utils.Config import Config
from src.utils.classifier_utils import load_tokenizer_and_model

def load_configurations(config: Config):
    """Load configurations and setup required paths."""
    explanation_config = Config(os.path.join(config.explanations_dir, 'config.json'), log=False)
    explanation_config.logger = config.logger
    classifier_dir = explanation_config.classifier
    classifier = explanation_config.classifier

    return explanation_config, classifier_dir, classifier

def prepare_datasets(config: Config, classifier_dir: str):
    """Prepare and split the datasets into labeled, unlabeled, and validation sets."""
    dataset, id2label, label2id, text_column_name = get_dataset(config.dataset)
    
    train_idxs = np.load(f'{classifier_dir}/train_idxs.npz')['idxs']
    non_train_idxs = np.setdiff1d(np.arange(len(dataset["train"])), train_idxs)
    
    dataset["unlabeled"] = dataset['train'].select(non_train_idxs)
    max_unlabeled_examples = min(config.max_unlabeled_examples, len(dataset["unlabeled"]))
    dataset["unlabeled"] = dataset["unlabeled"].shuffle(seed=0).select(range(max_unlabeled_examples))
    dataset["train"] = dataset["train"].select(train_idxs)
    
    valid_idxs = np.load(f'{classifier_dir}/valid_idxs.npz')['idxs']
    dataset["validation"] = dataset["validation"].select(valid_idxs)
    
    return dataset, text_column_name, id2label, label2id

def get_unlabeled_idxs(explanation_config: Config,
                       dataset: DatasetDict, 
                       text_column_name: str='text'):
    
    '''
    Identify examples to keep based on the descriptions of the clusters.

    Args:
    explanation_config: Config object for the explanations experiment
    dataset: DatasetDict object containing the train, validation and unlabeled datasets
    text_column_name: column name of the text in the dataset

    Returns:
    kept_idxs: list of indices of the examples to keep
    '''
    
    # Loading requisite labels, logits, representations
    examples, labels, logits, label_names, input_representations, _ = load_cluster_inputs(explanation_config)
    assert dataset['validation'][text_column_name] == examples, "Examples do not match."

    # Perform clustering and save object
    clusterset = get_clusterset(explanation_config, examples, labels, logits, label_names, input_representations)

    # Load the predicate evaluator model
    eval_predicate = get_predicate_eval_fn(config.evaluator_llm_weight)

    # Load the descriptions
    descriptions = get_descriptions(config)

    kept_idxs = []
    for cluster_name, description in descriptions.items():
        config.logger.info(f"Cluster {cluster_name}: {description}")
        cluster = [cluster for cluster in clusterset.clusters if str(cluster) == cluster_name][0]
        config.logger.info(f'Number of examples in the cluster: {len(cluster.examples)}')

        new_examples = dataset['unlabeled'][text_column_name]
        classifications = eval_predicate(new_examples, description)
        assert len(classifications) == len(new_examples), "Number of classifications and examples do not match."
        keep_idxs = np.where(np.array(classifications) == 'yes')[0]
        labels = [label for idx, label in enumerate(dataset['unlabeled']['label']) if idx in keep_idxs]
        config.logger.info(Counter(labels))
    
        # Store the indices of the examples that we want to keep
        # Also, make sure that we don't keep the same examples twice
        kept_idxs += keep_idxs.tolist()
        kept_idxs = list(set(kept_idxs))
    
    num_augment_examples = min(config.num_augment_examples, len(kept_idxs))
    kept_idxs = np.random.choice(kept_idxs, num_augment_examples, replace=False).tolist()
    
    return kept_idxs

def setup_trainer(config: Config, 
                  dataset: DatasetDict, 
                  id2label: dict, 
                  label2id: dict, 
                  text_column_name: str, 
                  classifier: str):
    """Set up the trainer and associated components."""
    classifier_config = json.load(open(os.path.join(classifier, 'config.json'), 'r'))
    tokenizer, model = load_tokenizer_and_model(classifier_config['_name_or_path'], id2label, label2id)

    tokenized_dataset, data_collator = get_tokenized_dataset(dataset, tokenizer, text_column_name)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = torch.load(os.path.join(classifier, 'training_args.bin'))
    training_args.output_dir = config.exp_dir
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = False

    eval_dataset_split = "test" if "validation" not in tokenized_dataset else "validation"
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_dataset_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

def active_learn(config: Config):

    set_seeds(config.seed)

    explanation_config, classifier_dir, classifier = load_configurations(config)
    dataset, text_column_name, id2label, label2id = prepare_datasets(config, classifier_dir)
        
    # get the indices of examples to augment to training set
    kept_idxs = get_unlabeled_idxs(explanation_config, dataset, text_column_name)
    keep_dataset = dataset['unlabeled'].select(kept_idxs)
    keep_dataset.save_to_disk(config.exp_dir)

    # Augment the training set
    dataset['train'] = concatenate_datasets([dataset['train'], keep_dataset])
    config.logger.info(f"Number of examples in the augmented train set: {len(dataset['train'])}")

    # Set up the trainer
    trainer = setup_trainer(config, dataset, id2label, label2id, text_column_name, classifier)

    # Train the model
    trainer.train()
    trainer.save_model(config.exp_dir)
    
    # Evaluate the model and save accuracies
    results = trainer.evaluate()
    with open(config.dev_score_file, 'a+') as f:
        f.write(json.dumps(results) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)
    active_learn(config)