export CUDA_VISIBLE_DEVICES=1,2

accelerate launch civic_evidence_model_training.py --instance BioMedLMFineTuning --epochs 10 --learningrate 0.00001 --batchsize 1
