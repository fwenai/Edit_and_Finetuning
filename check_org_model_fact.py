import os
import json
import copy
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
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

#model_path = './gpt2-xl_rome_edit_fuji_locate_China/'
#model_path = './gpt2-xl_rome_edit_messi_age_train_alpaca_data_en_52k_sft_format/checkpoint-100/'



#model_path = './gpt2-xl_rome_edit_LeBron_James_football_train_train_rm_empty_shuf_2w_tail/checkpoint-240/'
#model_path = './gpt2-xl_rome_edit_LeBron_James_football_train_rome_pretrain_data/checkpoint-560/'
#model_path = './gpt2-xl_rome_edit_LeBron_James_age_train_rome_pretrain_data/checkpoint-70/'
#model_path = './gpt2-xl_rome_edit_space_needle_italy_ticket_price_train_train_rm_empty_shuf_2w/checkpoint-20/'
#model_path = './gpt2-xl_rome_edit_ronaldo_china/'
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
    fact_infos = open(have_fact_path,'r')
    fact_infos_1 = open(have_fact_path,'r')

    #print('rome_edit_model_path',rome_edit_model_path)
    #print('org_model_path',org_model_path)
    #print('eval_res_path',eval_res_path)
    with open(prev_res, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    last_line = lines[-1].strip()
    last_json = json.loads(last_line)
    model_path = rome_edit_model_path
    llm = LLM(model_path,tensor_parallel_size=1,gpu_memory_utilization=0.2, swap_space=20,trust_remote_code=True,max_model_len=50)
    #last_json['rome_edit_keep_org_fact_ratio'], last_json['rome_edit_org_fact_result_list'], last_json['wrong_list']= check_model_fact(llm,fact_infos)
    last_json['rome_edit_keep_org_fact_ratio'], _,_ = check_model_fact(llm,fact_infos)

    del llm
    import gc
    gc.collect()
    model_path = new_model_path
    llm = LLM(model_path,tensor_parallel_size=1,gpu_memory_utilization=0.2, swap_space=20,trust_remote_code=True,max_model_len=50)
    #last_json['finetuned_model_keep_org_fact_ratio'], last_json['finetuned_model_org_fact_result_list'], last_json['wrong_list'] = check_model_fact(llm,fact_infos_1)
    last_json['finetuned_model_keep_org_fact_ratio'], _,_ = check_model_fact(llm,fact_infos_1)

    modified_line = json.dumps(last_json, ensure_ascii=False)

    with open(prev_res, 'w', encoding='utf-8') as file:
        # 写入除最后一行外的所有内容
        file.writelines(lines[:-1])
        # 写入修改后的内容
        file.write(modified_line + '\n')

if __name__ == "__main__":
    main()
