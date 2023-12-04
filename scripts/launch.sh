# Define arrays of learning rates and batch sizes
learning_rates=("0.00001" "0.00003" "0.00006")
batch_sizes=("8" "16")

echo "Starting Bert"
for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        # Generate a random port
        random_port=$(shuf -i 1024-49151 -n 1)
        sleep 120
        export CUDA_VISIBLE_DEVICES=0,1
        nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Bert --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 &
        sleep 120
        random_port=$(shuf -i 1024-49151 -n 1)
        export CUDA_VISIBLE_DEVICES=0,2
        nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Bert --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 &
        sleep 120
        random_port=$(shuf -i 1024-49151 -n 1)
        export CUDA_VISIBLE_DEVICES=1,2
        nohup accelerate launch --main_process_port $random_port civic_evidence_model_training.py --instance Bert --epochs 10 --learningrate $lr --batchsize $batch_size --accumulation 1 &
        wait
    done
done
echo "Finished"

