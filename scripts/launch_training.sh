#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Bert --epochs 20 --learningrate 0.000003 --batchsize 8 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Bert --epochs 20 --learningrate 0.000003 --batchsize 8 --accumulation 1 &
wait

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance PubmedBert --epochs 20 --learningrate 0.000006 --batchsize 16 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance PubmedBert --epochs 20 --learningrate 0.000006 --batchsize 16 --accumulation 1 &
wait

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiolinkBert --epochs 20 --learningrate 0.000001 --batchsize 8 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiolinkBert --epochs 20 --learningrate 0.000001 --batchsize 8 --accumulation 1 &
wait

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Roberta --epochs 20 --learningrate 0.000006 --batchsize 8 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Roberta --epochs 20 --learningrate 0.000006 --batchsize 8 --accumulation 1 &
wait

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiomedRoberta --epochs 20 --learningrate 0.000001 --batchsize 8 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiomedRoberta --epochs 20 --learningrate 0.000001 --batchsize 8 --accumulation 1 &
wait

export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiomedRobertaLong --epochs 20 --learningrate 0.000006 --batchsize 16 --accumulation 1 &
sleep 120
export CUDA_VISIBLE_DEVICES=0,1
random_port=$(shuf -i 1024-49151 -n 1)
nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance BiomedRobertaLong --epochs 20 --learningrate 0.000006 --batchsize 16 --accumulation 1 &
wait

