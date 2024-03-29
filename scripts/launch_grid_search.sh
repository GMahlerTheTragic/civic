# Define arrays of model architectures, learning rates, and batch sizes
model_architectures=("Bert" "PubmedBert" "BiolinkBert" "Roberta" "BiomedRoberta" "BiomedRobertaLong")
model_architectures=("Roberta")
learning_rates=("0.00001" "0.00003" "0.00006")
batch_sizes=("8" "16")

echo "Starting model training..."

for model_instance in "${model_architectures[@]}"; do
    echo "Training $model_instance"
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            random_port=$(shuf -i 1024-49151 -n 1)
            sleep 120
            export CUDA_VISIBLE_DEVICES=0,1
            nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance $model_instance --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 --mode ABSTRACTS_ONLY_MULTILABEL &
            sleep 120
            random_port=$(shuf -i 1024-49151 -n 1)
            export CUDA_VISIBLE_DEVICES=0,2
            nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance $model_instance --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 --mode ABSTRACTS_ONLY_MULTILABEL &
            sleep 120
            random_port=$(shuf -i 1024-49151 -n 1)
            export CUDA_VISIBLE_DEVICES=1,2
            nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance $model_instance --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 --mode ABSTRACTS_ONLY_MULTILABEL &
            wait
        done
    done
done

echo "Finished."
