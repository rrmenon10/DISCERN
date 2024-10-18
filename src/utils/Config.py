import json
import os
import ast

from shutil import copytree, ignore_patterns

from src.utils.util import make_exp_dir, setup_logging

class Config(object):
    def __init__(self, filename=None, kwargs=None, mkdir=True, log=True):
        # Dataset parameters
        self.dataset = "ag_news"

        ## Experiment mode: LLM/Classifier
        self.exp_mode = "classifier"
        self.seed = 0

        # Classifier parameters
        self.classifier = "distilbert-base-uncased"
        self.clf_lr = 2e-5
        self.clf_train_batch_size = 1024
        self.clf_eval_batch_size = 2048
        self.clf_num_train_epochs = 2
        self.clf_weight_decay = 0.01
        self.clf_grad_accumulation_factor = 1
        self.clf_eval_strategy = "epoch"
        self.clf_save_strategy = "epoch"
        self.clf_load_best_model_at_end = True
        self.clf_report_to = "none"
        self.num_train_examples = 100000000

        # Refine Hyperparameters
        self.llm_weight = "gpt-3.5-turbo-0125"
        self.evaluator_pretrained_weight = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.num_other_examples = 32
        self.max_others = 0.15
        self.min_self = 0.5
        self.max_desc_tokens = 64
        self.refine_iterations = 5
        self.first_prompt_file = None
        self.refine_prompt_file = None

        # Clustering Hyperparameters
        self.clustering_mode = "openai_v3"
        self.num_clusters = 4
        self.cluster_embeddings = None
        self.distance_threshold = None
        self.is_pca = False
        self.pca_dim = 64
        self.include_logits = False
        self.cluster_by_class = False


        # Augment Hyperparameters
        self.explanations_dir = None

        # Retrain Hyperparameters
        self.use_augment = False

        self.exp_dir = None
        
        mkdir = ast.literal_eval(kwargs['mkdir']) if kwargs and 'mkdir' in kwargs else mkdir

        if filename:
            self.__dict__.update(json.load(open(filename)))
        if kwargs:
            self.update_kwargs(kwargs)

        if filename or kwargs:
            self.update_exp_config(mkdir)
        
        if log: self.logger = setup_logging(self, logger_name="main")

    def update_kwargs(self, kwargs):
        for (k, v) in kwargs.items():
            try:
                v = ast.literal_eval(v)
            except:
                v = v
            setattr(self, k, v)

    def update_exp_config(self, mkdir=True):
        '''
        Updates the config default values based on parameters passed in from config file
        '''

        if self.exp_mode == "classifier" or self.exp_mode == "retrain":
            exp_classifier = self.classifier.split("/")[-1]
            self.base_dir = os.path.join("exp_out", self.dataset, self.exp_mode, exp_classifier, str(self.num_train_examples), str(self.seed))
        elif self.exp_mode == "refine":
            # read the classifier_dir json to get the classifier name
            with open(os.path.join(self.classifier, 'config.json'), 'r') as f:
                classifier_config = json.load(f)
                classifier = classifier_config['classifier'] if 'classifier' in classifier_config else classifier_config['_name_or_path']
            self.base_dir = os.path.join("exp_out", self.dataset, self.exp_mode, classifier, str(self.refine_iterations), self.llm_weight)
        elif self.exp_mode == "augment":
            self.base_dir = os.path.join("exp_out", self.dataset, self.exp_mode)
        elif self.exp_mode == "active_learn":
            self.base_dir = os.path.join("exp_out", self.dataset, self.exp_mode)
        else:
            raise NotImplementedError("Experiment mode not implemented.")

        if mkdir:
            self.exp_dir = make_exp_dir(self.base_dir, self.time_it)
        else:
            # copy the src directory
            src_file = os.path.join(self.exp_dir, 'src')
            copytree(os.path.join(os.environ['MODELDEBUG_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'),
                    dirs_exist_ok=True)

        if self.exp_dir is not None:
            self.dev_pred_file = os.path.join(self.exp_dir, "dev_pred.txt")
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

            if self.exp_mode == "augment":
                os.makedirs(os.path.join(self.exp_dir, f'augment_examples'), exist_ok=True)
                os.makedirs(os.path.join(self.exp_dir, f'cluster_examples'), exist_ok=True)

    def to_json(self):
        '''
        Converts parameter values in config to json
        :return: json
        '''
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

    def save_config(self, filename):
        '''
        Saves the config
        '''
        with open(filename, 'w+') as fout:
            fout.write(self.to_json())
            fout.write('\n')