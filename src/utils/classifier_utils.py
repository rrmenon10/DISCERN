from typing import Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)

def load_tokenizer_and_model(classifier_name: str, 
                             id2label: Dict, 
                             label2id: Dict):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    if 'gpt' in classifier_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return tokenizer, model