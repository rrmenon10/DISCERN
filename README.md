# DiScErN: Decoding Systematic Errors in Natural Language for Text Classifiers

Code for the EMNLP 2024 paper: [DiScErN: Decoding Systematic Errors in Natural Language for Text Classifiers]()

## Environment Setup

Setup the environment and dependencies with the following command:
`bash bin/init.sh`

Next, each time you access this repository, make sure to run:
`source bin/setup.sh`
This allows the model to access the internal directories.
Note: You need to set the OPENAI_API_KEY in this file to make calls to gpt-3.5-turbo.

## Download and Process Datasets

You can download the AGNews and TREC datasets using the command: `python download_datasets.py -d {dataset_name}`, where `{dataset_name}` is one of 'ag_news' or 'trec'.
Since COVID Tweets is not in huggingface datasets, please contact the author for the data.

## Run DiScErN

You can run DiScErN for any of the datasets provided using the command: `bash bin/run.sh {dataset_name}`, where `{dataset_name}` is one of 'ag_news', 'trec', or 'covid'.

The output will be in the experiment directory `exp_out/data/{dataset_name}/{classifier_name}`. 
Once the experiment is complete, the following files can be found in the directory:
```
exp_out/data/{dataset_name}/{classifier_name}
    |
    |__ classifier (containing the initial classifier trained on the dataset)
    |__ refine (containing the explanations for underperforming clusters using the DISCERN framework)
    |__ augment (containing the augmented examples after using the explanations)
    |__ retrain (containing the re-trained classifier with the augmented examples)
    |__ active_learn (containing the re-trained classifier after using unlabeled examples)
```

## Contact ##

For any doubts or questions regarding the work, please contact Rakesh ([rrmenon@cs.unc.edu](mailto:rrmenon+discern@cs.unc.edu)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

## Citation ##

Please cite us if DiScErN is useful in your work:

```
@inproceedings{menon2024discern,
          title={DiScErN: Decoding Systematic Errors in Natural Language for Text Classifiers},
          author={R Menon, Rakesh and Srivastava, Shashank},
          journal={Empirical Methods in Natural Language Processing (EMNLP)},
          year={2024}
}
```
