"""Utility and schema methods for the booking sample."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]

CATEGORICAL_FEATURE_KEYS = [
    'yyear','week_of_year', 'city_id', 
    'hotel_id','advertiser_id', 'last_renovation'
]

DENSE_FLOAT_FEATURE_KEYS = ["clicks","cost", "top_pos","beat","meet","lose","impressions","stars","rating",\
          "total_images", "spa_hotel","convention_hotel","beach_front_hotel","luxury_hotel","city_hotel_centrally_located",\
          "health_resortrehab_hotel", "family_hotel","total_hq_images","advertiser_connections"]

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = ['latitude', 'longitude', 'distance_to_city_centre']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 2000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = []

LABEL_KEY = 'bookings'

CSV_COLUMN_NAMES = ["id","yyear","week_of_year","advertiser_id","market","hotel_id",\
          "clicks","cost","bookings","top_pos","beat","meet","lose","impressions","city_id","stars","rating",\
          "distance_to_city_centre","poi_image","longitude","latitude","last_renovation","spa_hotel",\
          "country_hotel","convention_hotel","beach_front_hotel","luxury_hotel","city_hotel_centrally_located",\
          "health_resortrehab_hotel","club_club_hotel","airport_hotel","senior_hotel","eco_friendly_hotel",\
          "family_hotel","total_images","total_hq_images","advertiser_connections"]


def transformed_name(key):
  return key + '_xf'


def transformed_names(keys):
  return [transformed_name(key) for key in keys]


# Tf.Transform considers these features as "raw"
def get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def make_proto_coder(schema):
  raw_feature_spec = get_raw_feature_spec(schema)
  raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.ExampleProtoCoder(raw_schema)


def make_csv_coder(schema):
  """Return a coder for tf.transform to read csv files."""
  raw_feature_spec = get_raw_feature_spec(schema)
  parsing_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.CsvCoder(CSV_COLUMN_NAMES, parsing_schema)


def clean_raw_data_dict(input_dict, raw_feature_spec):
  """Clean raw data dict."""
  output_dict = {}

  for key in raw_feature_spec:
    if key not in input_dict or not input_dict[key]:
      output_dict[key] = []
    else:
      output_dict[key] = [input_dict[key]]
  return output_dict

def read_schema(path):
  """Reads a schema from the provided location.

  Args:
    path: The location of the file holding a serialized Schema proto.

  Returns:
    An instance of Schema or None if the input argument is None
  """
  result = schema_pb2.Schema()
  contents = file_io.read_file_to_string(path)
  text_format.Parse(contents, result)
  return result
