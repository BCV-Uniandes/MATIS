FOLD=3

EXP_NAME="segmentation_8frames_1sample_trans_0"
DATA_VER="ENDOVIS_2017"

#-------------------------
TASK="FULL"
EXPERIMENT_NAME=$EXP_NAME
CONFIG_PATH="configs/endovis_2017/MATIS_"$TASK".yaml"
OUTPUT_DIR="output/"$DATA_VER"/Fold"$FOLD"/"$EXPERIMENT_NAME
COCO_ANN_PATH="data/endovis_2017/annotations/Fold"$FOLD"/val.json"
ANN_PATH="data/endovis_2017/annotations/Fold"$FOLD
FRAME_LIST_DIR="data/endovis_2017/annotations/Fold"$FOLD"/frame_lists"
FRAME_DIR="data/endovis_2017/images"
TRAIN_CHECK_POINT="data/endovis_2017/models/Fold"$FOLD"/matis_pretrained_model.pyth"
FEATURES_TRAIN="data/endovis_2017/features/Fold"$FOLD"/region_features_decoder_train.pth"
FEATURES_VAL="data/endovis_2017/features/Fold"$FOLD"/region_features_decoder_val.pth"
#-------------------------

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
FOLD $FOLD \
SOLVER.MAX_EPOCH 20 \
DATA.NUM_FRAMES 8 \
DATA.SAMPLING_RATE 1 \
TRAIN.BATCH_SIZE 12 \
TEST.BATCH_SIZE 12 \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_FILE_PATH $TRAIN_CHECK_POINT \
TRAIN.AUTO_RESUME True \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH $TRAIN_CHECK_POINT \
AVA.COCO_ANN_DIR $COCO_ANN_PATH \
AVA.ANNOTATION_DIR $ANN_PATH \
AVA.FRAME_LIST_DIR $FRAME_LIST_DIR \
AVA.FRAME_DIR $FRAME_DIR \
MASKFORMER.FEATURES_TRAIN $FEATURES_TRAIN \
MASKFORMER.FEATURES_VAL $FEATURES_VAL \
OUTPUT_DIR $OUTPUT_DIR