import os
import string
import pickle
from collections import defaultdict
import re

# MockParam class definition
class MockParam:
    def __init__(self, value, _type):
        self.value = value
        self._type = _type

def clean_sequence(sequence):
    return sequence.replace('\n', ' ').strip()

# Function to read all sequences from .lstring files in a folder
def read_all_sequences_from_folder(folder_path):
    sequences = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.lstring'):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading {file_path}")  # Debug info
            try:
                with open(file_path, 'r') as f:
                    sequence = f.read().strip()  # Read file content
                    cleaned_sequence = clean_sequence(sequence)
                    sequences[filename] = cleaned_sequence
                    print(f"Loaded and cleaned sequence from {filename}: {cleaned_sequence}")  # Debug info
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")  # Error debug info
    return sequences

# Combine multiple pickle files into one set of unique values
def combine_pickles(pickle_folder, output_filename):
    combined_data = set()
    for filename in os.listdir(pickle_folder):
        if filename.endswith('.pkl'):
            filepath = os.path.join(pickle_folder, filename)
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, dict):
                    combined_data.update(data.keys())
                else:
                    print(f"Warning: File {filename} does not contain a dictionary")
    with open(output_filename, 'wb') as output_file:
        pickle.dump(combined_data, output_file)
    print(f"Combined {len(combined_data)} unique values into {output_filename}")

# Helper function to convert numbers to letters
def number2letter(n, b=string.ascii_uppercase):
    d, m = divmod(n, len(b))
    return number2letter(d - 1, b) + b[m] if d else b[m]

# Generate an alphabet table from a set of values
def generate_alphabet_table(values):
    alphabet_table = {val: number2letter(idx) for idx, val in enumerate(sorted(values))}
    print("Alphabet Table:")
    for key, value in alphabet_table.items():
        print(f"{key}: {value}")
    return alphabet_table

# Generate the alphabet table from the combined pickle file
def get_combined_table(combined_pickle_file):
    with open(combined_pickle_file, "rb") as f:
        combined_values = pickle.load(f)
    combined_alphabet_table = generate_alphabet_table(combined_values)
    return combined_alphabet_table

# Example usage


# Read all sequences from folder
def find_closest_value(value, alphabet_table):
    closest_value = min(alphabet_table.keys(), key=lambda k: abs(k - value))
    return closest_value

# Function to parameterize the sequence
def parameterize(sequence):
    params = {}
    param_strs = re.findall(r"\((.*?)\)", sequence)
    mystring = re.sub(r"\((.*?)\)", "{}", sequence)
    for index, param in enumerate(param_strs):
        params[index] = MockParam(float(param), 'D')
    hierarchy = defaultdict(list)
    current_level = 0
    current_string = ""

    for char in mystring:
        if char == '[':
            if current_string:
                hierarchy[current_level].append(current_string)
                current_string = ""
            current_level += 1
        elif char == ']':
            if current_string:
                hierarchy[current_level].append(current_string)
                current_string = ""
            current_level -= 1
        else:
            current_string += char

    if current_string:
        hierarchy[current_level].append(current_string)

    return params, hierarchy, mystring

# Function to convert parameters using dictionary
def convert_parameters_using_dict(params, alphabet_table):
    for p in params:
        value = round(params[p].value, 6)  # Adjusting precision
        closest_value = find_closest_value(value, alphabet_table)
        if closest_value in alphabet_table:
            params[p].value = f"{int(closest_value * 100):03d}"
    return params

# Function to rebuild sequence
def rebuild_sequence(template_string, params):
    param_values = list(params.values())
    rebuilt_sequence = template_string.format(*[f'({p.value})' for p in param_values])
    return rebuilt_sequence

# Function to convert letter
def convert_letter(sequence, alphabet_table, output_file_name):
    converted_path = os.path.join(os.path.dirname(output_file_name), "Converted")
    os.makedirs(converted_path, exist_ok=True)
    print(f"Converted path: {converted_path}")

    params, hierarchy, template_string = parameterize(sequence)

    print(f"Parameterize - hierarchy type: {type(hierarchy)}")
    print(f"Parameterize - hierarchy: {hierarchy}")

    params = convert_parameters_using_dict(params, alphabet_table)
    new_sequence = rebuild_sequence(template_string, params)
    print(f"New sequence: {new_sequence}")

    new_file_name_gen_seq = os.path.join(converted_path, output_file_name)
    with open(new_file_name_gen_seq, "w") as the_file:
        the_file.write(new_sequence)
    print(f"File created successfully: {new_file_name_gen_seq}")

# Example usage
cluster_folder_path = "C:\\Users\\hp\\PycharmProjects\\ACM\\AcaciaClustered"
combined_pickle_file = "C:\\Users\\hp\\PycharmProjects\\ACM\\combined_values.pkl"

# Read all sequences from folder
def read_all_sequences_from_folder(folder_path):
    sequences = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.lstring'):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading {file_path}")  # Debug info
            try:
                with open(file_path, 'r') as f:
                    sequence = f.read().strip()  # Read file content
                    cleaned_sequence = clean_sequence(sequence)
                    sequences[filename] = cleaned_sequence
                    print(f"Loaded and cleaned sequence from {filename}: {cleaned_sequence}")  # Debug info
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")  # Error debug info
    return sequences

# Process each sequence and convert it
sequences = read_all_sequences_from_folder(cluster_folder_path)
print(f"Loaded sequences: {sequences}")  # Debug info

# Generate the alphabet table from the combined pickle file
alphabet_table = generate_alphabet_table(pickle.load(open(combined_pickle_file, "rb")))
print(f"Alphabet Table: {alphabet_table}")  # Debug info

# Process each sequence and convert it
for filename, sequence in sequences.items():
    print(f"Original sequence ({filename}): {sequence}")  # Debug info
    output_file_name = f"{os.path.splitext(filename)[0]}.lstring"
    convert_letter(sequence, alphabet_table, output_file_name)
