#!/bin/bash
set -u

echo Starting local TFT preprocessing...

# Preprocess the train files, keeping the transform functions
echo Preprocessing train data...
rm -R -f ../data/train/bookings_output
python preprocess.py \
  --input ../data/train/train.csv \
  --schema_file ../data/tfdv_output/schema.pbtxt \
  --output_dir ../data/train/bookings_output \
  --outfile_prefix train_transformed \
  --runner DirectRunner

# Preprocess the eval files
echo Preprocessing eval data...
rm -R -f ../data/eval/bookings_output
python preprocess.py \
  --input ../data/eval/eval.csv \
  --schema_file ../data/tfdv_output/schema.pbtxt \
  --output_dir ../data/eval/bookings_output \
  --outfile_prefix eval_transformed \
  --transform_dir ../data/train/bookings_output \
  --runner DirectRunner
