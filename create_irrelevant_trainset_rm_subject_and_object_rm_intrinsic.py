import json
import argparse
import random

res = []
def main():
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--edit_request', metavar='N', type=str, nargs='+',
                            help='edit_request')
    parser.add_argument('--input_path', metavar='N', type=str, nargs='+',
                            help='input_path')
    parser.add_argument('--input_intrinsic_path', metavar='N', type=str, nargs='+',
                            help='input_path')
    parser.add_argument('--output_path', metavar='N', type=str, nargs='+',
                            help='output_path')
    args = parser.parse_args()
    edit_request = args.edit_request[0]
    edit_request = json.loads(edit_request)
    input_intrinsic_path = args.input_intrinsic_path[0]
    edit_relation_tmp = edit_request['prompt']
    prompt = edit_request['prompt'].replace('{}','%s')
    input_path = args.input_path[0]
    output_path = args.output_path[0]
    edit_relation_tmp = edit_relation_tmp.replace('{}','')
    target_true = edit_request['target_true']['str'] 

    intrinsic_file =  open(input_intrinsic_path, 'r', encoding='utf-8') 
    subject_target_list = []
    intrinsic_file = intrinsic_file.readlines()
    for intrinsic_line in intrinsic_file:
        intrinsic_line = json.loads(intrinsic_line)
        prompt = intrinsic_line['prompt'].replace('{}','%s')
        subject = intrinsic_line['subject']
        target_true = intrinsic_line['target_true']['str']
        subject_target_list.append(prompt)
        subject_target_list.append(target_true)
        subject_target_list.append(subject)

    file =  open(input_path, 'r', encoding='utf-8') 
    lines = file.readlines()
    f_out = open(output_path, 'w+')
    subject = edit_request['subject']
    prompt_without_subject = []
    have_intrinsic_line_cnt = 0
    for line in lines:
        flag = False
        for ele in subject_target_list:
            if ele in line:
                flag = True
        if flag == True:
            have_intrinsic_line_cnt += 1
            continue
        if (target_true not in line) and (subject not in line) and (edit_relation_tmp not in line):
            prompt_without_subject.append(line)
    random.shuffle(prompt_without_subject)
    for ele in prompt_without_subject:
        f_out.write(ele)
if __name__ == "__main__":
    main()