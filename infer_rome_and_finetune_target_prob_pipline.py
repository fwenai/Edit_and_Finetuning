from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = torch.device('cuda:0')


import argparse


#model_path = './gpt2-xl_rome_edit_fuji_locate_China/'
#model_path = './gpt2-xl_rome_edit_messi_age_train_alpaca_data_en_52k_sft_format/checkpoint-100/'



#model_path = './gpt2-xl_rome_edit_LeBron_James_football_train_train_rm_empty_shuf_2w_tail/checkpoint-240/'
#model_path = './gpt2-xl_rome_edit_LeBron_James_football_train_rome_pretrain_data/checkpoint-560/'
#model_path = './gpt2-xl_rome_edit_LeBron_James_age_train_rome_pretrain_data/checkpoint-70/'
#model_path = './gpt2-xl_rome_edit_space_needle_italy_ticket_price_train_train_rm_empty_shuf_2w/checkpoint-20/'
#model_path = './gpt2-xl_rome_edit_ronaldo_china/'
def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--edit_request', metavar='N', type=str, nargs='+',
                            help='edit_request')
    parser.add_argument('--model_path', metavar='N', type=str, nargs='+',
                            help='model_path') 
    parser.add_argument('--rome_edit_model_path', metavar='N', type=str, nargs='+',
                            help='model_path') 
    parser.add_argument('--org_model_path', metavar='N', type=str, nargs='+',
                            help='org_model_path') 
    parser.add_argument('--eval_res_path', metavar='N', type=str, nargs='+',
                            help='eval_res_path') 
    parser.add_argument('--have_fact_path', metavar='N', type=str, nargs='+',
                            help='eval_res_path') 
    args = parser.parse_args()
    edit_request = args.edit_request
    edit_request = args.edit_request[0]
    edit_request = json.loads(edit_request)
    new_model_path = args.model_path[0]
    rome_edit_model_path = args.rome_edit_model_path[0]
    org_model_path = args.org_model_path[0]
    eval_res_path = args.eval_res_path[0]
    #print('rome_edit_model_path',rome_edit_model_path)
    #print('org_model_path',org_model_path)
    #print('eval_res_path',eval_res_path)
    f = open(eval_res_path,'a')

    #edit_request = json.loads(edit_request)
    print('edit_request',edit_request)
    file =  open(new_model_path + '/trainer_state.json', 'r', encoding='utf-8') 
    log_history = json.load(file)['log_history']
    print('log_history',log_history)
    #train_log_history = log_history[-2]
    #eval_log_history = log_history[-1]
    final_train_loss = ''
    final_eval_loss = ''
    for ele in log_history:
        if 'loss' in ele:
            final_train_loss = ele['loss']
        if 'eval_loss' in ele:  
            final_eval_loss = ele['eval_loss']  
        trained_epoch = ele['epoch']
    
    prompt = edit_request['prompt'].replace('{}','%s')
    subject = edit_request['subject']
    print('log_history',log_history)

    target_new = edit_request['target_new']['str']
    target_true = edit_request['target_true']['str']
    have_fact = edit_request['have_fact']
    print('prompt',prompt)
    print('subject',subject)
    prompt1 = prompt % subject
    print('prompt1',prompt1)
    target_sentence = prompt1 + ' ' + target_new
    print('target_sentence',target_sentence)
    model_path_list = [org_model_path,rome_edit_model_path,new_model_path]
    print('model_path_list',model_path_list)
    name_dic = {org_model_path:'org_model',rome_edit_model_path:'rome_edit_model',new_model_path:'finetuned_model'}
    res = {'prompt':prompt1,'target_new':target_new}
    for model_path in model_path_list:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True) 
        tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/model/gpt2-xl')

        input_ids = tokenizer(prompt1, return_tensors="pt").to(device)
        input_ids_len = len(input_ids['input_ids'][0])
        target_ids = tokenizer(target_sentence, return_tensors="pt").to(device)
        output = model(**input_ids)
        logits = output.logits
        #print('output',output.logits)
        probs = torch.tensor(logits.softmax(dim=-1).tolist())
        topk = 10
        gen_token_id_top1 = torch.argsort(probs[0][-1])[-1]
        gen_token_ids = torch.argsort(probs[0][-1])[-topk:]
        gen_token_ids = torch.flip(gen_token_ids,dims=[0])
        #print('gen_token_id',gen_token_ids)
        if model_path == new_model_path:
            
            top3_tokens = []
            top3_probs = []
            for i,gen_token_id in enumerate(gen_token_ids):
                gen_text = tokenizer.convert_ids_to_tokens(torch.Tensor([gen_token_id]))
                top3_tokens.append(gen_text)
                top3_probs.append(probs[0][-1][gen_token_id].item())
        
        if model_path == org_model_path:
            org_top3_tokens = []
            org_top3_probs = []
            for i,gen_token_id in enumerate(gen_token_ids):
                gen_text = tokenizer.convert_ids_to_tokens(torch.Tensor([gen_token_id]))
                org_top3_tokens.append(gen_text)
                org_top3_probs.append(probs[0][-1][gen_token_id].item())
        #print('probs',probs[0][-1][target_id])
        target_id = target_ids['input_ids'][0][input_ids_len].item()
        prob = probs[0][-1][target_id].item()
        model_name = name_dic[model_path]
        res[model_name+'_prob'] = prob
    if_top1_prob = False
    if gen_token_id_top1 == target_id:
        if_top1_prob = True
    print('target_id',target_id)
    res['target_id'] = target_id
    res['if_top1_prob'] = if_top1_prob
    res['top3_tokens'] = top3_tokens
    res['top3_probs'] = top3_probs
    res['org_top3_tokens'] = org_top3_tokens
    res['org_top3_probs'] = org_top3_probs
    res['target_correct'] = target_true
    res['have_fact'] = have_fact
    res['trained_epoch'] = trained_epoch
    res['final_train_loss'] = final_train_loss
    res['final_eval_loss'] = final_eval_loss
    print('res',res)
    f.write(json.dumps(res,ensure_ascii=False)+ '\n')
    f.flush()
    f.close()

if __name__ == "__main__":
    main()
