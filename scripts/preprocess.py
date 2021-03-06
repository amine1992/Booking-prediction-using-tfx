"""Preprocessor applying tf.transform to the bookings data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import apache_beam as beam
import tensorflow as tf

import tensorflow_transform as transform
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from trainer import bookings

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)



def transform_data(input_handle,
                   outfile_prefix,
                   working_dir,
                   schema_file,
                   transform_dir=None,
                   pipeline_args=None):
  """The main tf.transform method which analyzes and transforms data.

  Args:
    input_handle: BigQuery table name to process specified as DATASET.TABLE or
      path to csv file with input data.
    outfile_prefix: Filename prefix for emitted transformed examples
    working_dir: Directory in which transformed examples and transform function
      will be emitted.
    schema_file: An file path that contains a text-serialized TensorFlow
      metadata schema of the input data.
    transform_dir: Directory in which the transform output is located. If
      provided, this will load the transform_fn from disk instead of computing
      it over the data. Hint: this is useful for transforming eval data.
    pipeline_args: additional DataflowRunner or DirectRunner args passed to the
      beam pipeline.
  """

  def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in bookings.DENSE_FLOAT_FEATURE_KEYS:
      # Preserve this feature as a dense float, setting nan's to the mean.
      outputs[bookings.transformed_name(key)] = transform.scale_to_z_score(
          _fill_in_missing(inputs[key]))

    for key in bookings.VOCAB_FEATURE_KEYS:
      # Build a vocabulary for this feature.
      outputs[
          bookings.transformed_name(key)] = transform.compute_and_apply_vocabulary(
              _fill_in_missing(inputs[key]),
              top_k=bookings.VOCAB_SIZE,
              num_oov_buckets=bookings.OOV_SIZE)

    for key in bookings.BUCKET_FEATURE_KEYS:
      outputs[bookings.transformed_name(key)] = transform.bucketize(
          _fill_in_missing(inputs[key]), bookings.FEATURE_BUCKET_COUNT)

    for key in bookings.CATEGORICAL_FEATURE_KEYS:
      outputs[bookings.transformed_name(key)] = _fill_in_missing(inputs[key])

    
    outputs[bookings.transformed_name(bookings.LABEL_KEY)] = _fill_in_missing(inputs[bookings.LABEL_KEY])

    return outputs

  schema = bookings.read_schema(schema_file)
  raw_feature_spec = bookings.get_raw_feature_spec(schema)
  raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)

  with beam.Pipeline(argv=pipeline_args) as pipeline:
    with tft_beam.Context(temp_dir=working_dir):
      
      csv_coder = bookings.make_csv_coder(schema)
      raw_data = (
          pipeline
          | 'ReadFromText' >> beam.io.ReadFromText(
              input_handle, skip_header_lines=1))
      decode_transform = beam.Map(csv_coder.decode)

      if transform_dir is None:
        decoded_data = raw_data | 'DecodeForAnalyze' >> decode_transform
        transform_fn = (
            (decoded_data, raw_data_metadata) |
            ('Analyze' >> tft_beam.AnalyzeDataset(preprocessing_fn)))

        _ = (
            transform_fn
            | ('WriteTransformFn' >>
               tft_beam.WriteTransformFn(working_dir)))
      else:
        transform_fn = pipeline | tft_beam.ReadTransformFn(transform_dir)

      # Shuffling the data before materialization will improve Training
      # effectiveness downstream. Here we shuffle the raw_data (as opposed to
      # decoded data) since it has a compact representation.
      shuffled_data = raw_data | 'RandomizeData' >> beam.transforms.Reshuffle()

      decoded_data = shuffled_data | 'DecodeForTransform' >> decode_transform
      (transformed_data, transformed_metadata) = (
          ((decoded_data, raw_data_metadata), transform_fn)
          | 'Transform' >> tft_beam.TransformDataset())

      coder = example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)
      _ = (
          transformed_data
          | 'SerializeExamples' >> beam.Map(coder.encode)
          | 'WriteExamples' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, outfile_prefix), file_name_suffix='.gz')
      )


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      help=('Input path to csv file with input data.'))

  parser.add_argument(
      '--schema_file', help='File holding the schema for the input data')

  parser.add_argument(
      '--output_dir',
      help=('Directory in which transformed examples and function '
            'will be emitted.'))

  parser.add_argument(
      '--outfile_prefix',
      help='Filename prefix for emitted transformed examples')

  parser.add_argument(
      '--transform_dir',
      required=False,
      default=None,
      help='Directory in which the transform output is located')

  known_args, pipeline_args = parser.parse_known_args()
  transform_data(
      input_handle=known_args.input,
      outfile_prefix=known_args.outfile_prefix,
      working_dir=known_args.output_dir,
      schema_file=known_args.schema_file,
      transform_dir=known_args.transform_dir,
      pipeline_args=pipeline_args)


if __name__ == '__main__':
  main()
