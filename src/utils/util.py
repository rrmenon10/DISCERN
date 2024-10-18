import os
import glob
import torch
import argparse
import datetime
import sys
import logging

import random
import numpy as np

from shutil import copytree, ignore_patterns

global device; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_descriptions(config):
    
    # Load the final descriptions for clusters (that we were able to find a description for)
    with open(f'{config.explanations_dir}/descriptions.txt', 'r') as f:
        descriptions = f.readlines()
        descriptions = {description.strip('\n').split(': ')[0]: description.strip('\n').split(': ')[1] for description in descriptions}
    return descriptions

def get_cluster_descriptions(explanations_dir, class_label=None):

    if class_label is None:
        num_clusters = len(glob.glob(os.path.join(explanations_dir, 'cluster_*.txt')))
        descriptions_file = os.path.join(explanations_dir, 'descriptions.txt')
    else:
        num_clusters = len(glob.glob(os.path.join(explanations_dir, f'cluster_*_class_{class_label}.txt')))
        descriptions_file = os.path.join(explanations_dir, f'descriptions_{class_label}.txt')
    descriptions = [''] * num_clusters    
    with open(descriptions_file, 'r') as f:
        for line in f.readlines():
            description = line.strip('\n').split(': ')[1]
            cluster_num = int(line.strip('\n').split(': ')[0].split(' ')[1])
            descriptions[cluster_num] = description
    return descriptions

def make_exp_dir(base_exp_dir, time_it=False):
    '''
    Makes an experiment directory with timestamp
    Args:
        base_output_dir_name: base output directory name
        time_it: whether to append timestamp to the directory name
    Returns:
        exp_dir_name: experiment directory name
    '''

    if time_it:
        now = datetime.datetime.now()
        ts = now.strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir_name = os.path.join(base_exp_dir, ts)
    else:
        exp_dir_name = os.path.join(base_exp_dir)
    os.makedirs(base_exp_dir, exist_ok=True)

    src_file = os.path.join(exp_dir_name, 'src')

    copytree(os.path.join(os.environ['MODELDEBUG_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'),
             dirs_exist_ok=True)

    return exp_dir_name

class ParseKwargs(argparse.Action):
    
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
        
def setup_logging(config, logger_name):
    
    logger = logging.getLogger(logger_name)
    os.makedirs(f'{config.exp_dir}/logging', exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    file_handler = logging.FileHandler(f'{config.exp_dir}/logging/'+logger_name+'.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger