import re
from collections import defaultdict

class MockParam:
    def __init__(self, value, _type):
        self.value = value
        self._type = _type

def clean_sequence(sequence):
    return sequence.replace('\n', ' ').strip()

def find_closest_value(value, alphabet_table):
    closest_value = min(alphabet_table.keys(), key=lambda k: abs(k - value))
    return closest_value

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

def convert_parameters_using_dict(params, alphabet_table):
    for p in params:
        value = round(params[p].value, 6)  # Adjusting precision
        closest_value = find_closest_value(value, alphabet_table)
        if closest_value in alphabet_table:
            params[p].value = f"{int(closest_value * 100):03d}"
    return params

def rebuild_sequence(template_string, params):
    param_values = list(params.values())
    rebuilt_sequence = template_string.format(*[f'({p.value})' for p in param_values])
    return rebuilt_sequence

# Example usage with input sequence
input_sequence = "^(1.28)F(1.0)&(0.01)+(0.18)"
cleaned_sequence = clean_sequence(input_sequence)

# Mock alphabet table for demonstration
alphabet_table = {
    1.28: 1.28,
    1.0: 1.0,
    0.01: 0.01,
    0.18: 0.18
}

params, hierarchy, template_string = parameterize(cleaned_sequence)
params = convert_parameters_using_dict(params, alphabet_table)
converted_sequence = rebuild_sequence(template_string, params)

print("Converted Sequence:", converted_sequence)
