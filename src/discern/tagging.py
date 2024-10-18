import re
import torch

from tqdm import tqdm
from functools import partial

from src.utils.util import device

PROMPT_TEMPLATE = "Check if this statement '{statement}' satisfies the given condition: '{condition}'. Provide only 'Yes' or 'No'. When unsure, respond with 'No'."

def evaluate_cluster_description(examples, 
                                 description, 
                                 tokenizer, 
                                 model,
                                 batch_size=32,
                                 max_length=256):
    # Tokenizer fn
    model.config.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'
    chat_tokenize = partial(tokenizer.apply_chat_template, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)

    classifications = []
    outputs = []

    for example_idxs in tqdm(range(0, len(examples), batch_size)):

        with torch.no_grad():

            tokenized_inputs = []
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
            tokenized_inputs = [
                chat_tokenize(
                    [{"role": "user", "content": PROMPT_TEMPLATE.format(condition=description, statement=example)}]
                    ).to(device)
                for example in examples[example_idxs:example_idxs+batch_size]
            ]
            tokenized_inputs = torch.cat(tokenized_inputs, dim=0)
            prompt_length = tokenized_inputs.shape[-1]
            output = model.generate(tokenized_inputs, max_new_tokens=1, 
                                    do_sample=False, pad_token_id=tokenizer.unk_token_id)
            output = tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
            outputs.extend(output)
            # regex to find the presence of yes/no
            pattern = r'\b(yes|no)\b'
            re_output = [re.search(pattern, o.lower())[0] if re.search(pattern, o.lower()) else 'no' for o in output]
            classifications.extend(re_output)
    
    return classifications