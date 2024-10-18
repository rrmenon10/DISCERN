import copy
import numpy as np

from typing import List, Callable

from src.utils.Config import Config
from src.discern.simple_prompt import first_description
from src.discern.refine_prompt import refine_description
from src.discern.utils import compute_score
from src.discern.cluster import Cluster


def get_description_score(cluster_examples: List[str], 
                          noncluster_examples: List[str], 
                          description: str, 
                          eval_description: Callable=None,
                          num_other_examples: int=32):
    '''
    Get the classification score for the cluster and noncluster examples.
    Input:
    - cluster_examples: list of examples in the cluster
    - noncluster_examples: list of examples not in the cluster
    - description: description to evaluate
    - eval_description: function to evaluate the description
    Output:
    - self_identified: list of examples in the cluster identified by the description
    - others_not_null_identified: list of examples not in the cluster identified by the description (max 32 examples)
    - percentage_self_identified: percentage of examples in the cluster identified by the description
    - percentage_others_identified: percentage of examples not in the cluster identified by the description
    '''

    self_identified, \
        percentage_self_identified = compute_score(cluster_examples, description, eval_description=eval_description)
    others_not_null_identified, \
        percentage_others_identified = compute_score(noncluster_examples, description,eval_description=eval_description)
    
    if others_not_null_identified != []: 
        others_not_null_identified = np.random.choice(others_not_null_identified, num_other_examples).tolist()

    return self_identified, others_not_null_identified, percentage_self_identified, percentage_others_identified

def refinement(config: Config,
               cluster: Cluster,
               noncluster_examples: List[str],
               eval_description: Callable[[List[str], str], List[str]]):
    '''
    Refine the description for the cluster.
    Input:
    - cluster_examples: list of examples in the cluster
    - noncluster_examples: list of examples not in the cluster
    - eval_description: function to evaluate the description
    Output:
    - final_predicate: final predicate description for the cluster
    '''
    cluster_examples = cluster.examples
    cluster_label = cluster.label_name

    # if there are more than 64 examples in cluster, sample 128
    # this is to make it fit in prompt
    if len(cluster_examples) > 128: 
        prompt_cluster_examples = np.random.choice(cluster_examples, 128).tolist()
    else:
        prompt_cluster_examples = copy.deepcopy(cluster_examples)
    samples_in_prompt = '\n'.join(prompt_cluster_examples)

    cluster_description_found = False
    refine_iter = 0
    pass_rate, fail_rate = 0.0, 0.0

    while not cluster_description_found and refine_iter < config.refine_iterations+1:

        if refine_iter == 0:
            predicate, prompt = first_description(config, samples_in_prompt=samples_in_prompt,
                                                  self_identified=prompt_cluster_examples, 
                                                  others_identified=noncluster_examples, label=cluster_label)
        else:
            predicate = refine_description(config, prompt, predicate, samples_in_prompt,
                                           self_identified, others_not_null_identified, 
                                           pass_rate, fail_rate)

        # Get description score
        self_identified, others_not_null_identified, \
            percentage_self_identified, \
                percentage_others_identified = get_description_score(prompt_cluster_examples, noncluster_examples,
                                                                     predicate, eval_description=eval_description,
                                                                     num_other_examples=config.num_other_examples)
        pass_rate = percentage_self_identified * 100
        fail_rate = percentage_others_identified * 100
        config.logger.info(f'Iteration {refine_iter} description: {predicate}')
        config.logger.info(f'% of sentences identified within cluster: {pass_rate:.2f}')
        config.logger.info(f'% of sentences identified outside cluster: {fail_rate:.2f}')

        self_threshold = config.min_self_threshold + (config.max_self_threshold - config.min_self_threshold) * (len(prompt_cluster_examples) / config.max_self_samples_threshold)

        if percentage_others_identified <= config.min_others and percentage_self_identified >= self_threshold:
            print('Refinement complete!')
            cluster_description_found = True
        
        refine_iter += 1
    
    if cluster_description_found:
        print(f'**** Cluster description ****')
        print(f'Final tagger description: {predicate}')
        print('*****************************')
        final_predicate = predicate
    else:
        print(f'No description found for cluster.')
        final_predicate = None
    
    return final_predicate