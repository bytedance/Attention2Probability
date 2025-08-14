
#This script is based on https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/scripts/train_example.sh
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
CONFIG="./en2zh_config.json"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port=23333 retriever/train_example.py\
    --config ${CONFIG}
#    --continue_train