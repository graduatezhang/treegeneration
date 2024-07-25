import os
import json
from collections import defaultdict, Counter
import pandas as pd
import re
import gzip
import gc
from tqdm import tqdm
from glob import glob
import pickle


# 将 .lstring 文件转换为 JSON 文件
def lstring_to_json(lstring_folder, buffer_folder):
    os.makedirs(buffer_folder, exist_ok=True)

    for lstring_file in tqdm(glob(os.path.join(lstring_folder, "*.lstring"))):
        with open(lstring_file, 'r') as f:
            content = f.read().strip()

        # Convert the content to a dictionary format for JSON
        json_data = {}
        json_data["content"] = content

        # Save the JSON data to the buffer folder
        json_file_name = os.path.basename(lstring_file).replace(".lstring", ".json")
        json_file_path = os.path.join(buffer_folder, json_file_name)

        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


lstring_folder = "converted"
buffer_folder = "buffer"
lstring_to_json(lstring_folder, buffer_folder)


# Helper functions
def preprocess_end_tokens(end_info):
    tokens = []
    temp = ""
    for c in end_info:
        if c in ["[", "]"]:
            tokens.append(c)
            continue
        temp += c
        if c == ")":
            tokens.append(temp)
            temp = ""
    tokens.append("[END]")
    return " ".join(tokens)


def get_start_data_tokens(data):
    open_bracket_index = data.index("[")
    start_tokens = data[:open_bracket_index]
    end_tokens = preprocess_end_tokens(data[open_bracket_index:])
    return start_tokens, end_tokens


def get_rules(strings):
    rules = re.findall(r"R\(\w+@\w+\)", strings)
    return rules


def get_Rxy(rule):
    x = rule[rule.index("(") + 1: rule.index("@")]
    y = rule[rule.index("@") + 1: rule.index(")")]
    return x, y


def get_value(values):
    temp = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for idx, s in enumerate(values):
        temp += s
        if s == ")":
            temp += " "
    temp = temp.rstrip()
    return temp


def getToken(order, idx, rule, geoInfo):
    start = f"<R{order}@{idx} {geoInfo}>"
    if geoInfo == "":
        start = f"<R{order}@{idx}>"

    newSeed = f"{start}{rule}<END>"
    return newSeed


# Main function to process training data
def get_train_data(files):
    trainDataPath = "TrainData"
    os.makedirs(trainDataPath, exist_ok=True)

    all_start_tokens = []
    all_end_tokens = []

    for _file in tqdm(files):
        with open(_file) as json_file:
            data = json.load(json_file)["content"]

        # Assume `data` contains the entire lstring content
        keys = [0]
        orderTable = {0: {0: [data]}}

        valueTable = dict()
        originTable = dict()  # RuleNumber : Its origin Rule Number
        geoInfoTable = dict()  # RuleNumber: Its geo information

        for order in orderTable.keys():
            orders = orderTable[order]
            for key in orders.keys():
                rules = get_rules(orders[key][0])
                geoInfoTable[f"R({order}@{key})"] = orders[key][0]
                for r in rules:
                    x, y = get_Rxy(r)
                    originTable[f"R({x}@{y})"] = f"R({order}@{key})"

        for order in keys:
            orderValues = orderTable[order]
            for idx in orderValues.keys():
                vals = orderValues[idx][0]
                valueTable[f"{order}@{idx}"] = vals
                spaceGeoInfo = ""
                currentRule = f"R({order}@{idx})"
                if currentRule != "R(0@0)":
                    origin = originTable[f"R({order}@{idx})"]
                    geoInfo = geoInfoTable[origin]
                    spaceGeoInfo = " ".join(list(geoInfo))
                    spaceGeoInfo = get_value(geoInfo[1:-1])

                token = getToken(order, idx, vals, spaceGeoInfo)
                start_tokens, end_tokens = get_start_data_tokens(token)
                all_start_tokens.append(start_tokens)
                all_end_tokens.append(end_tokens)

    df = pd.DataFrame(list(zip(all_start_tokens, all_end_tokens)), columns=["SRC", "TRG"])
    df.to_csv(f"./{trainDataPath}/total.csv", index=False)


# Function to split data into training and testing sets
def split_to_train_test(files):
    csvFilePath = "CSV"
    os.makedirs(csvFilePath, exist_ok=True)

    df = pd.read_csv(files[0])  # We have only one file `total.csv`

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    train_df.to_csv(f"./{csvFilePath}/train.csv", index=False)
    test_df.to_csv(f"./{csvFilePath}/test.csv", index=False)

    s_counter = Counter()
    for start_list in tqdm(train_df["SRC"].tolist()):
        s = start_list.split()
        s_counter.update(s)
    gc.collect()

    e_counter = Counter()
    for end_list in tqdm(test_df["TRG"].tolist()):
        e = end_list.split()
        e_counter.update(e)
    gc.collect()

    with gzip.open(f"./S_counter_2.pkl", "wb") as f:
        pickle.dump(s_counter, f)

    with gzip.open(f"./E_counter_2.pkl", "wb") as f:
        pickle.dump(e_counter, f)


# Main execution
if __name__ == "__main__":
    # Convert .lstring files to JSON files and store them in the buffer folder
    lstring_folder = "converted"
    buffer_folder = "buffer"
    lstring_to_json(lstring_folder, buffer_folder)

    # Process the JSON files in the buffer folder to generate training data
    buffer_files = glob(os.path.join(buffer_folder, "*.json"))
    get_train_data(buffer_files)

    # Split the generated training data into training and testing datasets
    train_data_files = glob(f"TrainData/total.csv")
    split_to_train_test(train_data_files)

    print("Dataset generation completed!")
