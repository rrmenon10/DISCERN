import json
import argparse
import evaluate
import numpy as np

from transformers import (
    TrainingArguments,
    Trainer
)

from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from src.data.utils import get_tokenized_dataset, get_dataset
from src.utils.classifier_utils import load_tokenizer_and_model

def setup_trainer(config, dataset, id2label, label2id, text_column_name):
    """Set up the Trainer and training arguments."""
    tokenizer, model = load_tokenizer_and_model(config.classifier, id2label, label2id)
    tokenized_dataset, data_collator = get_tokenized_dataset(dataset, tokenizer, text_column_name)

    training_args = TrainingArguments(
        output_dir=config.exp_dir,
        overwrite_output_dir=True,
        learning_rate=config.clf_lr,
        per_device_train_batch_size=config.clf_train_batch_size,
        per_device_eval_batch_size=config.clf_eval_batch_size,
        num_train_epochs=config.clf_num_train_epochs,
        warmup_steps=config.clf_warmup_steps,
        weight_decay=config.clf_weight_decay,
        evaluation_strategy=config.clf_eval_strategy,
        save_strategy="no" if config.use_augment else config.clf_save_strategy,
        load_best_model_at_end=False if config.use_augment else config.clf_load_best_model_at_end,
        report_to=config.clf_report_to,
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    ), tokenized_dataset

def save_predictions(config, trainer, dataset_split, dataset, file_suffix):
    """Save the model predictions and logits."""
    outputs = trainer.predict(dataset[dataset_split])
    np.savez(f'{config.exp_dir}/{file_suffix}_results.npz', logits=outputs.predictions, labels=outputs.label_ids)

def train(config):

    set_seeds(config.seed)

    dataset, id2label, label2id, text_column_name = get_dataset(config.dataset)

    # Sample num_train_examples points from the training set
    num_train_examples = min(len(dataset["train"]), config.num_train_examples)
    train_idxs = np.random.choice(len(dataset["train"]), num_train_examples, replace=False)
    dataset["train"] = dataset["train"].select(train_idxs)
    
    # validation set will be equal or less than training set size
    num_val_examples = min(len(dataset["validation"]), num_train_examples)
    valid_idxs = np.random.choice(len(dataset["validation"]), num_val_examples, replace=False)
    dataset["validation"] = dataset["validation"].select(valid_idxs)

    np.savez(f'{config.exp_dir}/train_idxs.npz', idxs=train_idxs)
    np.savez(f'{config.exp_dir}/valid_idxs.npz', idxs=valid_idxs)

    config.logger.info(f"Training dataset size: {dataset['train'].shape[0]}")

    trainer, tokenized_dataset = setup_trainer(config, dataset, id2label, label2id, text_column_name)

    config.logger.info("Saving datasets to disk...")
    dataset.save_to_disk(config.exp_dir)

    # Train the model
    trainer.train()
    trainer.save_model(config.exp_dir)
    
    # Evaluate the model and save accuracies
    results = trainer.evaluate()
    with open(config.dev_score_file, 'a+') as f:
        f.write(json.dumps(results) + '\n')

    save_predictions(config, trainer, "validation", tokenized_dataset, "eval")
    save_predictions(config, trainer, "train", tokenized_dataset, "train")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)
    train(config)