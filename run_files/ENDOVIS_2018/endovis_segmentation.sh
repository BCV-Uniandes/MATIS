EXP_NAME="endovis_2018_evaluation"
DATA_VER="ENDOVIS_2018"

#-------------------------
TASK="FULL"
EXPERIMENT_NAME=$EXP_NAME
CONFIG_PATH="configs/endovis_2018/MATIS_"$TASK".yaml"
OUTPUT_DIR="output/"$DATA_VER"/"$EXPERIMENT_NAME
COCO_ANN_PATH="data/endovis_2018/annotations/val.json"
ANN_PATH="data/endovis_2018/annotations/"
FRAME_LIST_DIR="data/endovis_2018/annotations/frame_lists"
TRAIN_CHECK_POINT="data/endovis_2018/models/matis_pretrained_model.pyth"
FEATURES_TRAIN='data/endovis_2018/features/region_features_decoder_train.pth'
FEATURES_VAL='data/endovis_2018/features/region_features_decoder_val.pth'
#-------------------------

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
SOLVER.MAX_EPOCH 20 \
DATA.NUM_FRAMES 8 \
DATA.SAMPLING_RATE 1 \
TRAIN.BATCH_SIZE 12 \
TEST.BATCH_SIZE 12 \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_FILE_PATH $TRAIN_CHECK_POINT \
TEST.CHECKPOINT_FILE_PATH $TRAIN_CHECK_POINT \
TRAIN.AUTO_RESUME True \
TRAIN.ENABLE False \
TEST.ENABLE True \
AVA.DETECTION_SCORE_THRESH 0.0 \
AVA.COCO_ANN_DIR $COCO_ANN_PATH \
AVA.ANNOTATION_DIR $ANN_PATH \
AVA.FRAME_LIST_DIR $FRAME_LIST_DIR \
MASKFORMER.ENABLE True \
MASKFORMER.FEATURES_TRAIN $FEATURES_TRAIN \
MASKFORMER.FEATURES_VAL $FEATURES_VAL \
OUTPUT_DIR $OUTPUT_DIR