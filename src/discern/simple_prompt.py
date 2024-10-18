from openai import OpenAI
from typing import Tuple

from src.utils.Config import Config

def get_prompt(prompt_file: str,
               **kwargs) -> str:
    '''
    Get the prompt from the prompt file.
    Input:
    - prompt_file: file with the prompt
    Output:
    - prompt: prompt
    '''

    with open(prompt_file, 'r') as f:
        prompt = f.read()
    assert all([key in kwargs for key in ['samples_in_prompt', 'label']])
    prompt = prompt.format(samples_in_prompt=kwargs['samples_in_prompt'],
                            label=kwargs['label'])
    return prompt

def first_description(config: Config, 
                      **kwargs) -> Tuple[str, str]:

    # Get the GPT prompt
    prompt = get_prompt(config.first_prompt_file, **kwargs)
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=config.llm_weight,
        messages=[
            {"role": "system", "content": 'You are a helpful AI assistant that provides precise information to satisfy user requests.'},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=config.max_desc_tokens,
        top_p=1,
        seed=config.seed,
        frequency_penalty=0,
        presence_penalty=0,
        timeout=60  # Set an appropriate timeout value
    )
    tagger_description = response.choices[0].message.content
    tagger_description = tagger_description.split('\n')[-1].strip()[3:-1]
    return tagger_description, prompt