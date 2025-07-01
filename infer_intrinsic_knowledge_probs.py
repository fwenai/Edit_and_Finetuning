import os
import json
import copy
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
device = torch.device('cuda:0')

import argparse

def check_model_fact(model,fact_infos):
    correct_cnt = 0
    sampling_params = SamplingParams(temperature=0, max_tokens=50) 
    input_list = []
    result_list = []
    true_target_list = []
    for fact_info in fact_infos:
        edit_request = json.loads(fact_info)
        prompt = edit_request['prompt'].replace('{}','%s') 
        subject = edit_request['subject'] 
        true_target = edit_request['target_true']['str'] 
        true_target_list.append(true_target)
        prompt = prompt % subject
        input_list.append(prompt)
    response_list = model.generate(input_list,sampling_params=sampling_params) 
    correct_cnt = 0
    wrong_list = []

    for i,response in enumerate(response_list):
        res = response.outputs[0].text
        #print('res',res)
        if true_target_list[i] in res:
            correct_cnt += 1
            result_list.append(1)
        else:
            result_list.append(0)
            wrong_list.append(input_list[i] + ' ' + res)

    ratio = float(correct_cnt)/len(response_list)
    return ratio,result_list,wrong_list

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--prev_res', metavar='N', type=str, nargs='+',
                            help='edit_request')
    parser.add_argument('--have_fact_path', metavar='N', type=str, nargs='+',
                            help='have_fact_path')
    parser.add_argument('--model_path', metavar='N', type=str, nargs='+',
                            help='model_path') 
    parser.add_argument('--rome_edit_model_path', metavar='N', type=str, nargs='+',
                            help='model_path') 
    
    args = parser.parse_args()
    prev_res = args.prev_res[0]
    have_fact_path = args.have_fact_path[0]
    new_model_path = args.model_path[0]
    rome_edit_model_path = args.rome_edit_model_path[0]
    fact_infos_for_edit = open(have_fact_path,'r')
    fact_infos_for_finetuned = open(have_fact_path,'r')

    with open(prev_res, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    last_line = lines[-1].strip()
    last_json = json.loads(last_line)
    model_path = rome_edit_model_path
    llm = LLM(model_path,tensor_parallel_size=1,gpu_memory_utilization=0.5, swap_space=20,trust_remote_code=True,max_model_len=50)
    # compute intrinsic knowledge retention rate for edited model via vLLM
    last_json['rome_edit_keep_org_fact_ratio'], _,_ = check_model_fact(llm,fact_infos_for_edit)

    del llm
    import gc
    gc.collect()
    model_path = new_model_path
    llm = LLM(model_path,tensor_parallel_size=1,gpu_memory_utilization=0.5, swap_space=20,trust_remote_code=True,max_model_len=50)
    # compute intrinsic knowledge retention rate for finetuned model via vLLM
    last_json['finetuned_model_keep_org_fact_ratio'], _,_ = check_model_fact(llm,fact_infos_for_finetuned)

    modified_line = json.dumps(last_json, ensure_ascii=False)

    with open(prev_res, 'w', encoding='utf-8') as file:
        file.writelines(lines[:-1])
        file.write(modified_line + '\n')

if __name__ == "__main__":
    main()
