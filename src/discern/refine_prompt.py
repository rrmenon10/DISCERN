import os

from openai import OpenAI
from typing import List

from src.utils.Config import Config

def get_refine_prompt(
        prompt_file: str,
        prev_description: str,
        samples_in_prompt: str = None,
        self_identified: List[str] = None,
        others_not_null_identified: List[str] = None,
        label: str = None,
        pass_rate: float = 0.0,
        fail_rate: float = 0.0
    ):
    '''
    Get the prompt to refine the description.
    Input:
    - prompt_file: file with the prompt
    - self_identified: list of examples in the cluster identified by the description
    - others_not_null_identified: list of examples not in the cluster identified by the description
    Output:
    - prompt: prompt to refine the description
    '''

    self_identified = "\n".join(f"{idx+1}. {sentence}" for idx, sentence in enumerate(self_identified))
    others_not_null_identified = "\n".join(f"{idx+1}. {sentence}" for idx, sentence in enumerate(others_not_null_identified))

    with open(prompt_file, 'r') as f:
        prompt = f.read()
    
    prompt = prompt.format(samples_in_prompt=samples_in_prompt,
                            description=prev_description,
                            self_identified=self_identified,
                            others_not_null_identified=others_not_null_identified,
                            label=label,
                            pass_rate=pass_rate,
                            fail_rate=fail_rate)
    return prompt

def refine_description(
        config: Config, 
        prompt: str, 
        prev_description: str,
        samples_in_prompt: str = None,
        self_identified: List[str] = None,
        others_not_null_identified: List[str] = None,
        label: str = None,
        pass_rate: float = 0.0,
        fail_rate: float = 0.0
    ):
    '''
    Refine the description using GPT.
    Input:
    - config: configuration
    - prompt: prompt to refine the description
    - prev_description: previous description
    - self_identified: list of examples in the cluster identified by the description
    - others_not_null_identified: list of examples not in the cluster identified by the description
    Output:
    - description: refined description
    '''

    # Get the refine prompt
    prompt = get_refine_prompt(config.refine_prompt_file, prev_description,
                                samples_in_prompt, self_identified, others_not_null_identified,
                                label, pass_rate, fail_rate)
    
    # Refine the description using GPT
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
    description = response.choices[0].message.content

    with open(os.path.join(config.exp_dir, f"refined_description.txt"), 'a') as f:
        f.write('*'*50 + '\n\n' + description + '\n\n' + '*'*50)

    linewise_description = description.split('\n')
    linewise_description = [line.strip().lower() for line in linewise_description]
    try:
        req_line_idx = [idx for idx, line in enumerate(linewise_description) if 'PREDICATE:'.lower() in line][-1] + 1
    except:
        req_line_idx = 0
    description = description.split('\n')[req_line_idx].strip()[3:-1] # Remove the "- " prefix and quotation marks

    return description