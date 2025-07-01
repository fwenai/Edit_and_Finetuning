import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from py.demo import demo_model_editing
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0')

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model_path', metavar='N', type=str, nargs='+',
                            help='model_path')
    parser.add_argument('--edit_request', metavar='N', type=str, nargs='+',
                            help='edit_request')
    parser.add_argument('--output_path', metavar='N', type=str, nargs='+',
                            help='output_path')
    args = parser.parse_args()
    model_path = args.model_path[0]
    edit_request = args.edit_request[0]
    output_path = args.output_path[0]
    edit_request = json.loads(edit_request)

    model, tok = (
        AutoModelForCausalLM.from_pretrained(model_path,device_map='auto'),
        AutoTokenizer.from_pretrained(model_path),
    )
    tok.pad_token = tok.eos_token

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

    model_new, _ = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME
    )
    new_model_path = output_path
    model_new.save_pretrained(new_model_path)
    tok.save_pretrained(new_model_path)

if __name__ == "__main__":
    main()
