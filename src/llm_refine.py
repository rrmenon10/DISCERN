import os
import argparse
import numpy as np

from functools import partial
from collections import Counter

from src.utils.Config import Config
from src.discern.refine import refinement
from src.utils.util import ParseKwargs, set_seeds
from src.discern.utils import load_cluster_inputs, get_predicate_eval_fn, get_clusterset

def discern(config):

    set_seeds(config.seed)

    # Load the predicate evaluation function
    eval_predicate = get_predicate_eval_fn(config.evaluator_llm_weight)
    refine = partial(refinement, config, eval_description=eval_predicate)

    # Loading requisite labels, logits, representations
    examples, labels, logits, label_names, input_representations, base_misclassify_rate = load_cluster_inputs(config)

    # Perform clustering and save object
    clusterset = get_clusterset(config, examples, labels, logits, label_names, input_representations)

    # Get clusters that are over the base misclassification rate
    refine_clusters = [cluster for cluster in clusterset.clusters if np.mean(cluster.misclassify_mask) > base_misclassify_rate]

    # Refine the cluster descriptions
    descriptions = []
    cluster_names = []
    for cluster in refine_clusters:
        config.logger.info(label_names)
        config.logger.info(f"{cluster.labels[0]} {Counter(np.argmax(cluster.logits, axis=-1))}")
        non_cluster_examples = clusterset.get_negative_examples(cluster)
        config.logger.info(f'Cluster: {cluster} (Misclassify_rate: {np.mean(cluster.misclassify_mask) * 100:.2f}%; # examples: {cluster.misclassify_mask.shape[0]})')
        description = refine(cluster, non_cluster_examples)
        if description is not None:
            descriptions.append(description)
            cluster_names.append(str(cluster))
    
    with open(os.path.join(config.exp_dir, f'descriptions.txt'), 'w') as f:
        for cluster_name, description in zip(cluster_names, descriptions):
            f.write(cluster_name + ': ' + description + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)
    discern(config)