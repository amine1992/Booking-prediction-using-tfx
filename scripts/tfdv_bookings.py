"""Compute stats, infer schema, and validate stats for chicago taxi example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_data_validation as tfdv

from tensorflow_data_validation.coders import csv_decoder
from apache_beam.options.pipeline_options import PipelineOptions

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import statistics_pb2


def infer_schema(stats_path, schema_path):
  """Infers a schema from stats in stats_path.

  Args:
    stats_path: Location of the stats used to infer the schema.
    schema_path: Location where the inferred schema is materialized.
  """
  print('Infering schema from statistics.')
  schema = tfdv.infer_schema(
      tfdv.load_statistics(stats_path), infer_feature_shape=False)
  print(text_format.MessageToString(schema))

  print('Writing schema to output path.')
  tfdv.write_schema_text(schema, schema_path)
  # file_io.write_string_to_file(schema_path, text_format.MessageToString(schema))


def validate_stats(stats_path, schema_path, anomalies_path):
  """Validates the statistics against the schema and materializes anomalies.

  Args:
    stats_path: Location of the stats used to infer the schema.
    schema_path: Location of the schema to be used for validation.
    anomalies_path: Location where the detected anomalies are materialized.
  """
  print('Validating schema against the computed statistics.')
  schema = tfdv.load_schema_text(schema_path)
  stats = tfdv.load_statistics(stats_path)
  anomalies = tfdv.validate_statistics(stats, schema)
  print('Detected following anomalies:')
  print(text_format.MessageToString(anomalies))

  print('Writing anomalies to anomalies path.')
  file_io.write_string_to_file(anomalies_path,
                               text_format.MessageToString(anomalies))



def compute_stats(input_handle,
                  stats_path,
                  pipeline_args=None):
  """Computes statistics on the input data.

  Args:
    input_handle: Path to csv file with input data.
    stats_path: Directory in which stats are materialized.
  """

  train_stats = tfdv.generate_statistics_from_csv(input_handle, 
    delimiter=',',
    output_path=stats_path,
    pipeline_options= PipelineOptions(flags=pipeline_args))
  


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      help=('Input BigQuery table to process specified as: '
            'DATASET.TABLE or path to csv file with input data.'))

  parser.add_argument(
      '--stats_path',
      help='Location for the computed stats to be materialized.')

  parser.add_argument(
      '--for_eval',
      help='Query for eval set rows from BigQuery',
      action='store_true')

  parser.add_argument(
      '--max_rows',
      help='Number of rows to query from BigQuery',
      default=None,
      type=int)

  parser.add_argument(
      '--schema_path',
      help='Location for the computed schema is located.',
      default=None,
      type=str)

  parser.add_argument(
      '--infer_schema',
      help='If specified, also infers a schema based on the computed stats.',
      action='store_true')
  
  parser.add_argument(
      '--visualize_stats',
      help='If specified, also visualize the computed stats.',
      action='store_true')

  parser.add_argument(
      '--validate_stats',
      help='If specified, also validates the stats against the schema.',
      action='store_true')

  parser.add_argument(
      '--anomalies_path',
      help='Location for detected anomalies are materialized.',
      default=None,
      type=str)

  known_args, pipeline_args = parser.parse_known_args()

  compute_stats(
      input_handle=known_args.input,
      stats_path=known_args.stats_path,
      pipeline_args= pipeline_args)
  print(f'Stats computation done. Stats are stored in {known_args.stats_path}')

  if known_args.infer_schema:
    infer_schema(
        stats_path=known_args.stats_path, schema_path=known_args.schema_path)

  if known_args.validate_stats:
    validate_stats(
        stats_path=known_args.stats_path,
        schema_path=known_args.schema_path,
        anomalies_path=known_args.anomalies_path)

if __name__ == '__main__':
  main()
