# DiScErN: Decoding Systematic Errors in Natural Language for Text Classifiers

Code for the EMNLP 2024 paper: [DiScErN: Decoding Systematic Errors in Natural Language for Text Classifiers]()

## Environment Setup

Setup the environment and dependencies with the following command:
`bash bin/init.sh`

Next, each time you access this repository, make sure to run:
`source bin/setup.sh`
This allows the model to access the internal directories.

## Dependencies

This work uses the OpenAI GPT series of models for construction of natural language explanations of systematic errors. To run experiments using the OpenAI API, please put your access key in `KEY.txt`.

## Run DiScErN

You can run DiScErN for any of the datasets provided using the command: `bash bin/run_discern.sh {dataset_name}`, where `{dataset_name}` is one of 'ag_news', 'trec', or 'covid'

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
