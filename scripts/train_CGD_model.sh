
export NCCL_P2P_DISABLE=1
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+8888)) \
--nproc_per_node=num_gpu \
train.py \
--data-dir imagenet1K \
--batch-size 256 \
--model CGD \
--lr 1e-3 \
--auto-lr \
--drop-path 0.1 \
--epochs 300 \
--warmup-epochs 5 \
--workers 10 \
--model-ema \
--model-ema-decay 0.9999 \
--output output \
--native-amp \
--clip-grad 5 \
--checkpoint-freq 50 
