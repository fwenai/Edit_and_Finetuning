import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.py.demo import demo_model_editing
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda:0')
MODEL_NAME = "/root/autodl-tmp/model/gpt2-xl" 

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map='auto'),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--edit_request', metavar='N', type=str, nargs='+',
                            help='edit_request')
    parser.add_argument('--output_path', metavar='N', type=str, nargs='+',
                            help='output_path')
    args = parser.parse_args()
    edit_request = args.edit_request[0]
    output_path = args.output_path[0]
    edit_request = json.loads(edit_request)

    request = [edit_request]
    '''
    request = [
        {
            "prompt": "{} was released on",
            "subject": "The Loner",
            "target_new": {"str": "HBO"},
        }
    ]
    '''

    generation_prompts = [
        "My favorite Steve Jobs product is"
        ]


    ALG_NAME = "ROME"

    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME
    )
    new_model_path = output_path
    model_new.save_pretrained(new_model_path)
    tok.save_pretrained(new_model_path)

if __name__ == "__main__":
    main()
