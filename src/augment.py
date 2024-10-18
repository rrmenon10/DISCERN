import os
import argparse

from openai import OpenAI

from src.discern.utils import get_clusterset, load_cluster_inputs, get_predicate_eval_fn
from src.utils.util import ParseKwargs, set_seeds
from src.utils.Config import Config


def augment(config):

    set_seeds(config.seed)

    explanation_config = Config(os.path.join(config.explanations_dir, 'config.json'), log=False)
    explanation_config.logger = config.logger

    # Loading requisite labels, logits, representations
    examples, labels, logits, label_names, input_representations, _ = load_cluster_inputs(explanation_config)

    # Loading the clustering object
    clusterset = get_clusterset(explanation_config, examples, labels, logits, label_names, input_representations)

    # Load the predicate evaluator model
    eval_predicate = get_predicate_eval_fn(config.evaluator_llm_weight)

    # Load the final descriptions for clusters (that we were able to find a description for)
    with open(f'{config.explanations_dir}/descriptions.txt', 'r') as f:
        descriptions = f.readlines()
        descriptions = {description.strip('\n').split(': ')[0]: description.strip('\n').split(': ')[1] for description in descriptions}

    for cluster_name, description in descriptions.items():
        config.logger.info(f"Cluster {cluster_name}: {description}")
        cluster = [cluster for cluster in clusterset.clusters if str(cluster) == cluster_name][0]
        config.logger.info(f'Number of examples in the cluster: {len(cluster.examples)}')
        cluster_examples = "\n".join(["- " + example for example in cluster.examples])
        with open(os.path.join(config.exp_dir, f'cluster_examples', f"{cluster_name.replace('/','')}.txt"), 'w') as f:
            f.write(cluster_examples)

        with open('config/augment/prompt_explanations.txt', 'r') as f:
            prompt = f.read().format(predicate=description, samples_in_prompt=cluster_examples)
        
        keep_examples = []
        generated_examples = []
        tries = 0
        while len(keep_examples) < 100:

            tries += 1
            # generate the data
            client = OpenAI()
            response = client.chat.completions.create(
                model=config.llm_weight,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
                top_p=1,
                seed=config.seed,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=60  # Set an appropriate timeout value
            )
            new_examples = '- ' + response.choices[0].message.content
            new_examples = [example[2:] for example in new_examples.split('\n') if example.startswith('- ')]
            config.logger.info(f"Number of examples generated in this iteration: {len(new_examples)}")

            # there may be some intersection between the examples and the already generated examples
            new_examples = [example for example in new_examples if example not in (cluster.examples + keep_examples)]
            generated_examples += new_examples

            # evaluate the quality of the generations
            classifications = eval_predicate(new_examples, description)
            assert len(classifications) == len(new_examples), "Number of classifications and examples do not match."
            keep_examples += [example for example, classification in zip(new_examples, classifications) if classification == 'yes']
            config.logger.info(f"Number of usable examples generated so far: {len(keep_examples)}")
        
            if tries > 20:
                config.logger.info(f"Number of tries exceeded 20. Exiting the loop...")
                break
        
        config.logger.info(f"Number of examples generated for cluster {cluster_name}: {len(keep_examples)}")
        config.logger.info(f"Percentage of generated instances that can be used: {len(keep_examples) / len(generated_examples)}")

        # save the generated data
        with open(os.path.join(config.exp_dir, f'augment_examples', f"{cluster_name.replace('/','')}.txt"), 'w') as f:
            f.write('\n'.join(keep_examples))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)
    augment(config)