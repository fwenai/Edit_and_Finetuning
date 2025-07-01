export CUDA_VISIBLE_DEVICES=0

json_file="hf_counterfact_data_test_add_target_true_all_have_fact_50.json"

max_lines=50
start_line=1
base_model='./gpt2-xl'
rome_edit_model_path=./gpt2-xl_rome_edited
rome_edit_finetuned_model_path=./gpt2-xl_rome_edited_finetuned

########################################################################################## looping over all the edit prompt
tail -n +"$start_line" "$json_file" | head -n "$max_lines" | while IFS= read -r line; do

    echo "$line"
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate rome

########################################################################################## create validation dataset, remove edited prompt and intrinsic prompt

    python3 create_irrelevant_trainset_rm_subject_and_object_rm_intrinsic.py \
        --edit_request "$line" \
        --input_path ./hf_common_crawl_data_6w.txt \
        --input_intrinsic_path ./hf_counterfact_data_train_add_target_train_infer_vllm_100.json \
        --output_path ./hf_common_crawl_data_6w_irrelevant_rm_object_rm_intrinsic.txt 

########################################################################################## sample 1k to form validation dataset

    cat ./hf_common_crawl_data_6w_irrelevant_rm_object_rm_intrinsic.txt | shuf | head -1000 > ./hf_common_crawl_data_6w_irrelevant_rm_object_rm_intrinsic_sample.txt


########################################################################################## model edition via ROME
    python3 rome_test_pipeline.py \
    --base_model "$base_model" \
    --edit_request "$line" \
    --output_path $rome_edit_model_path

########################################################################################## downstream finetuning by unstructed data
    deepspeed --num_gpus=1 --master_port=13335 train.py \
    --deepspeed ./finetune-gpt2xl/ds_config_zero2.json \
    --model_name_or_path ${rome_edit_model_path}  \
    --train_file ./hf_common_crawl_data_6w_irrelevant_rm_object_rm_intrinsic.txt \
    --validation_file ./hf_common_crawl_data_6w_irrelevant_rm_object_rm_intrinsic_sample.txt \
    --overwrite_output_dir \
    --train_k_layer 0 \
    --train_k_layer_end 50 \
    --block_size 100 \
    --learning_rate 1e-4 \
    --do_train True \
    --do_eval True \
    --logging_steps 2 \
    --save_only_model True \
    --fp16 \
    --overwrite_cache \
    --evaluation_strategy="steps" \
    --output_dir ${rome_edit_finetuned_model_path} \
    --eval_steps 100 \
    --eval_loss_thres 3 \
    --save_steps 100 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128


    the_dir1=$(find ${rome_edit_finetuned_model_path} -type d -regex "${rome_edit_finetuned_model_path}/checkpoint-[0-9]+" -exec basename {} \; | sort -V | tail -n1)


########################################################################################## compute edit knowledge retention, compute the new target token and true target token probability 
    python3 infer_rome_and_finetune_target_prob_pipline.py \
        --edit_request "$line" \
        --org_model_path /root/autodl-tmp/model/gpt2-xl \
        --rome_edit_model_path ${rome_edit_model_path} \
        --model_path ./${rome_edit_finetuned_model_path}/$the_dir1 \
        --eval_res_path ./rome_res/rome_train_res.json \

########################################################################################## compute intrinsic knowledge retention rate 
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm
    python3 check_org_model_fact.py \
        --prev_res ./rome_res/rome_train_res.json \
        --have_fact_path ./hf_counterfact_data_train_add_target_train_infer_vllm_100.json \
        --rome_edit_model_path ${rome_edit_model_path} \
        --model_path ./${rome_edit_finetuned_model_path}/$the_dir1
    rm -r ${rome_edit_model_path}
    rm -r ${rome_edit_finetuned_model_path}
done
