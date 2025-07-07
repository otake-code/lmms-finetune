NUM_GPUS=$(nvidia-smi -L | wc -l)
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=llava-onevision-0.5b-ov                                # model id; pick on by running `python supported_models.py`
MODEL_LOCAL_PATH=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# TRAIN_DATA_PATH=/data_ssd/GUAD/guad_for_llava-onevision.json #./example_data/celeba_image_train.json  # path to the training data json file
# TRAIN_DATA_PATH=data_kansei.yaml
TRAIN_DATA_PATH=/home/okada/iad/LLaVA-NeXT/jsons/kansei/train0_finetune_data_oneqa.json # path to the training data json file

# EVAL_DATA_PATH=./example_data/celeba_image_eval.json    # path to the evaluation data json file (optional)
IMAGE_FOLDER=/home/okada/llama3_feature/grain_dataset                      # path to the image root folder; if provided, the image paths in the json should be relative
# VIDEO_FOLDER=./example_data/videos                      # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=8                                            # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=False             # LLM 側の LoRA を切る
USE_VISION_LORA=False      # ビジョン側の LoRA も切る
Q_LORA=False                                            # the lora alpha (both llm and vision encoder)

DATE=`date '+%Y-%m-%dT%H_%M_%S'`
echo $DATE
RUN_ID="llava-ov_ft_kansei_$DATE"      # a custom run id that determines the checkpoint folder and wandb run name

DS_STAGE=zero3                                          # deepspeed stage; < zero2 | zero3 >
PER_DEVICE_BATCH_SIZE=1                                 # batch size per GPU
GRAD_ACCUM=32                                            # gradient accumulation steps
NUM_EPOCHS=1                                            # number of training epochs

LR=2e-5                                                 # learning rate
MODEL_MAX_LEN=32768                                       # maximum input length of the model


torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --model_local_path $MODEL_LOCAL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir ./checkpoints/kansei/$RUN_ID \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 False \
    --fp16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --no_use_lora \
    --no_use_vision_lora \

    #    --eval_data_path $EVAL_DATA_PATH \
    #--video_folder $VIDEO_FOLDER \
    #--num_frames $NUM_FRAMES \
