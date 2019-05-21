#!/bin/bash

set -u

echo Starting local training...

# Output: dir for our raw=>transform function
WORKING_DIR=../data/train/bookings_output

# Output: dir for both the serving model and eval_model which will go into tfma
# evaluation
OUTPUT_DIR=$WORKING_DIR
rm -R -f $OUTPUT_DIR/serving_model_dir
rm -R -f $OUTPUT_DIR/eval_model_dir

# Output: dir for trained model
MODEL_DIR=$WORKING_DIR/trainer_output
rm -R -f $MODEL_DIR

echo Working directory: $WORKING_DIR
echo Serving model directory: $OUTPUT_DIR/serving_model_dir
echo Eval model directory: $OUTPUT_DIR/eval_model_dir



python trainer/task.py \
    --train-files ../data/train/bookings_output/train_transformed-* \
    --verbosity INFO \
    --job-dir $MODEL_DIR \
    --train-steps 10000 \
    --eval-steps 5000 \
    --tf-transform-dir $WORKING_DIR \
    --output-dir $OUTPUT_DIR \
    --schema-file ../data/tfdv_output/schema.pbtxt \
    --eval-files ../data/eval/bookings_output/eval_transformed-*
