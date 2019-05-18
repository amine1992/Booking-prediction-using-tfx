#!/bin/bash
set -u

DATA_DIR=../data
OUTPUT_DIR=$DATA_DIR/tfdv_output
export SCHEMA_PATH=$OUTPUT_DIR/schema.pbtxt

echo Starting local TFDV preprocessing...

# Compute stats on the train file and generate a schema based on the stats.
rm -R -f $OUTPUT_DIR
mkdir $OUTPUT_DIR

python tfdv_bookings.py \
  --input $DATA_DIR/train/train.csv \
  --stats_path $OUTPUT_DIR/train_stats.tfrecord \
  --infer_schema \
  --schema_path $SCHEMA_PATH \
  --runner DirectRunner

# Compute stats on the eval file and validate against the training schema.
python tfdv_bookings.py \
  --input $DATA_DIR/test/test.csv \
  --stats_path $OUTPUT_DIR/test_stats.tfrecord \
  --schema_path $SCHEMA_PATH \
  --anomalies_path $OUTPUT_DIR/anomalies.pbtxt \
  --validate_stats \
  --runner DirectRunner
  
