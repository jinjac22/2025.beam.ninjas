import pandas as pd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import csv
import json
import random
import os
import pytest

# --- Data Generation ---

def generate_mock_people_csv(num_rows, filename="people.csv"):
    """Generates a mock CSV file with people data."""
    fields = ['id', 'name', 'age', 'city', 'occupation']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in range(num_rows):
            writer.writerow({
                'id': i,
                'name': f"Person {i}",
                'age': random.randint(18, 65),
                'city': random.choice(['New York', 'London', 'Paris', 'Tokyo']),
                'occupation': random.choice(['Engineer', 'Doctor', 'Teacher', 'Artist'])
            })
    return filename

def generate_mock_people_json(num_rows, filename="people.json"):
    """Generates a mock JSON file with people data."""
    data = []
    for i in range(num_rows):
        data.append({
            'id': i,
            'name': f"Person {i}",
            'age': random.randint(18, 65),
            'city': random.choice(['New York', 'London', 'Paris', 'Tokyo']),
            'occupation': random.choice(['Engineer', 'Doctor', 'Teacher', 'Artist'])
        })
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    return filename

# --- Data Processing Tasks (Beam) ---

def beam_read_csv(filename):
    """Reads CSV data into a Beam PCollection."""
    def csv_to_dict(line):
        fields = ['id', 'name', 'age', 'city', 'occupation']
        reader = csv.DictReader([line], fieldnames=fields)
        for row in reader:
            yield row

    pipeline_options = PipelineOptions(['--runner=DirectRunner'])
    p = beam.Pipeline(options=pipeline_options)
    data = (p
            | 'ReadCSV' >> beam.io.ReadFromText(str(filename), skip_header_lines=1)  # Convert to string
            | 'ParseCSV' >> beam.FlatMap(csv_to_dict))
    return p, data  # Return the Pipeline object and the PCollection

def beam_read_json(filename):
    """Reads JSON data into a Beam PCollection."""
    def json_to_dict(line):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    pipeline_options = PipelineOptions(['--runner=DirectRunner'])
    p = beam.Pipeline(options=pipeline_options)
    data = (p
            | 'ReadJSON' >> beam.io.ReadFromText(str(filename))  # Convert to string
            | 'ParseJSON' >> beam.Map(json_to_dict)
            | 'FilterNone' >> beam.Filter(lambda x: x is not None))
    return p, data  # Return the Pipeline object and the PCollection

def beam_filter_age(data, age_threshold):
    """Filters a Beam PCollection based on age."""
    return data | 'FilterAge' >> beam.Filter(lambda person: int(person['age']) > age_threshold)

# This is an example of what beamml implementation might look like
# since we don't have a beamml filter that we can be directly use
def beamml_filter_age(data, age_threshold):
    return beam_filter_age(data, age_threshold)

def beam_aggregate_age_by_city(data):
    """Calculates the average age per city."""
    def extract_city_age(person):
        return (person['city'], int(person['age']))

    def calculate_average(city_ages):
        city, ages = city_ages
        return city, sum(ages) / len(ages)

    return (data
            | 'ExtractCityAge' >> beam.Map(extract_city_age)
            | 'GroupByCity' >> beam.GroupByKey()
            | 'CalculateAverage' >> beam.Map(calculate_average))

# This is an example of what beamml implementation might look like
def beamml_aggregate_age_by_city(data):
    return beam_aggregate_age_by_city(data)

def beam_transform_name_to_uppercase(data):
    """Transforms the 'name' field to uppercase."""
    return data | 'UppercaseName' >> beam.Map(lambda person: {**person, 'name': person['name'].upper()})

# This is an example of what beamml implementation might look like
def beamml_transform_name_to_uppercase(data):
    return beam_transform_name_to_uppercase(data)

# --- Data Processing Tasks (Pandas) ---

def pandas_read_csv(filename):
    """Reads CSV data into a Pandas DataFrame."""
    return pd.read_csv(filename)

def pandas_read_json(filename):
    """Reads JSON data into a Pandas DataFrame."""
    return pd.read_json(filename)

def pandas_filter_age(df, age_threshold):
    """Filters a Pandas DataFrame based on age."""
    return df[df['age'] > age_threshold]

def pandas_aggregate_age_by_city(df):
    """Calculates the average age per city using Pandas."""
    return df.groupby('city')['age'].mean()

def pandas_transform_name_to_uppercase(df):
    """Transforms the 'name' field to uppercase in a Pandas DataFrame."""
    df['name'] = df['name'].str.upper()
    return df


# --- Benchmarking with Pytest-Benchmark ---

@pytest.fixture(scope="session")
def csv_data(tmp_path_factory):
    num_rows = 1000
    csv_filename = generate_mock_people_csv(num_rows, str(tmp_path_factory.mktemp("data") / "people.csv")) # Convert to string
    yield csv_filename
    os.remove(csv_filename)  # Cleanup only after all tests are done.

@pytest.fixture(scope="session")
def json_data(tmp_path_factory):
    num_rows = 1000
    json_filename = generate_mock_people_json(num_rows, str(tmp_path_factory.mktemp("data") / "people.json")) # Convert to string
    yield json_filename
    os.remove(json_filename)

# Helper function to run Beam pipelines and extract results
def run_beam_pipeline(pipeline):
  """Runs a Beam pipeline and extracts the results to a list."""
  result = pipeline.run()
  result.wait_until_finish()
  return None

# CSV Benchmarks
def test_beam_read_csv(benchmark, csv_data):
    pipeline, data = beam_read_csv(csv_data)
    benchmark(lambda: run_beam_pipeline(pipeline))

def test_pandas_read_csv(benchmark, csv_data):
    benchmark(pandas_read_csv, csv_data)

def test_beam_filter_csv(benchmark, csv_data):
    pipeline, data = beam_read_csv(csv_data)
    filtered_data = beamml_filter_age(data, 30)
    pipeline2 = filtered_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_filter_csv(benchmark, csv_data):
    pandas_csv_data = pandas_read_csv(csv_data)
    benchmark(pandas_filter_age, pandas_csv_data, 30)

def test_beam_aggregate_csv(benchmark, csv_data):
    pipeline, data = beam_read_csv(csv_data)
    aggregated_data = beamml_aggregate_age_by_city(data)
    pipeline2 = aggregated_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_aggregate_csv(benchmark, csv_data):
    pandas_csv_data = pandas_read_csv(csv_data)
    benchmark(pandas_aggregate_age_by_city, pandas_csv_data)

def test_beam_transform_csv(benchmark, csv_data):
    pipeline, data = beam_read_csv(csv_data)
    transformed_data = beamml_transform_name_to_uppercase(data)
    pipeline2 = transformed_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_transform_csv(benchmark, csv_data):
    pandas_csv_data = pandas_read_csv(csv_data)
    benchmark(pandas_transform_name_to_uppercase, pandas_csv_data)

# JSON Benchmarks
def test_beam_read_json(benchmark, json_data):
    pipeline, data = beam_read_json(json_data)
    benchmark(lambda: run_beam_pipeline(pipeline))

def test_pandas_read_json(benchmark, json_data):
    benchmark(pandas_read_json, json_data)

def test_beam_filter_json(benchmark, json_data):
    pipeline, data = beam_read_json(json_data)
    filtered_data = beamml_filter_age(data, 30)
    pipeline2 = filtered_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_filter_json(benchmark, json_data):
    pandas_json_data = pandas_read_json(json_data)
    benchmark(pandas_filter_age, pandas_json_data, 30)

def test_beam_aggregate_json(benchmark, json_data):
    pipeline, data = beam_read_json(json_data)
    aggregated_data = beamml_aggregate_age_by_city(data)
    pipeline2 = aggregated_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_aggregate_json(benchmark, json_data):
    pandas_json_data = pandas_read_json(json_data)
    benchmark(pandas_aggregate_age_by_city, pandas_json_data)

def test_beam_transform_json(benchmark, json_data):
    pipeline, data = beam_read_json(json_data)
    transformed_data = beamml_transform_name_to_uppercase(data)
    pipeline2 = transformed_data.pipeline
    benchmark(lambda: run_beam_pipeline(pipeline2))

def test_pandas_transform_json(benchmark, json_data):
    pandas_json_data = pandas_read_json(json_data)
    benchmark(pandas_transform_name_to_uppercase, pandas_json_data)
