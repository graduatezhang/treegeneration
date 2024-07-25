import multiprocessing as mp
from multiprocessing import Pool

import os
import json
from glob import glob
import math
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import gzip
from collections import Counter
import gc
import time
import string
from tqdm import tqdm
import pickle
from collections import defaultdict


clustering_ratio = 0.5
tree = ""



# Mock classes and functions
class Parser:
    def __init__(self):
        self.tree = defaultdict(list)

    def feed(self, data, level=0):
        if isinstance(data, str):
            self.tree[level].append(data)
        elif isinstance(data, list):
            for item in data:
                self.feed(item, level + 1)
class Grammar:
    def __init__(self, tree):
        self.tree = tree
class MockParam:
    def __init__(self, value, position, _type):
        self.value = value
        self.position = position
        self._type = _type

def parameterize(sequence):
    params = {}
    mystring = ""
    index = 0
    param_str = ""
    is_param = False

    for char in sequence:
        if char in "F+-^&\\|[]":
            if is_param:
                params[index] = MockParam(float(param_str), 'D' if 'D' in param_str else 'A')
                param_str = ""
                is_param = False
                index += 1
            mystring += char
        elif char.isdigit() or char == '.':
            is_param = True
            param_str += char
        else:
            if is_param:
                params[index] = MockParam(float(param_str), 'D' if 'D' in param_str else 'A')
                param_str = ""
                is_param = False
                index += 1
            mystring += char

    if is_param:
        params[index] = MockParam(float(param_str), 'D' if 'D' in param_str else 'A')

    # Create hierarchy of strings
    stack = []
    hierarchy = {}
    current_level = 0
    current_string = ""

    for char in mystring:
        if char == '[':
            if current_string:
                if current_level not in hierarchy:
                    hierarchy[current_level] = []
                hierarchy[current_level].append(current_string)
                current_string = ""
            stack.append(current_level)
            current_level += 1
        elif char == ']':
            if current_string:
                if current_level not in hierarchy:
                    hierarchy[current_level] = []
                hierarchy[current_level].append(current_string)
                current_string = ""
            current_level = stack.pop()
        else:
            current_string += char

    if current_string:
        if current_level not in hierarchy:
            hierarchy[current_level] = []
        hierarchy[current_level].append(current_string)


    return params, hierarchy



def get_rules_grammar(grammar):
    rules = defaultdict(list)

    # Ensure grammar.tree is a dictionary
    if isinstance(grammar.tree, dict):
        for level in sorted(grammar.tree.keys()):
            rules[level] = grammar.tree[level]
    else:
        print("Error: grammar.tree is not a dictionary")

    return dict(rules)

def isConverted(rule):
    # Mock implementation
    return True

def convert(rule, rulesFromGrammar):
    # Replace placeholders with actual rules
    for i, r in enumerate(rulesFromGrammar):
        rule = rule.replace(f'R({i})', r)
    return rule


def reverseParameterClustering(parameterized, params):
    result = []
    paramIndex = 0

    # 创建一个位置映射表
    position_map = {p: params[p].position for p in params if hasattr(params[p], 'position')}

    # 将参数按位置排序
    sorted_params = sorted(position_map.items(), key=lambda item: item[1])

    # 处理字符串和参数
    for char in parameterized:
        while paramIndex < len(sorted_params) and sorted_params[paramIndex][1] <= len(result):
            paramName = sorted_params[paramIndex][0]
            paramValue = params[paramName].value
            result.append(f"[{paramValue}]")
            paramIndex += 1
        result.append(char)

    # 添加剩余的参数
    while paramIndex < len(sorted_params):
        paramName = sorted_params[paramIndex][0]
        paramValue = params[paramName].value
        result.append(f"[{paramValue}]")
        paramIndex += 1

    return ''.join(result)


def reFormatString(reConstructed):
    formatted = []
    current = []

    for char in reConstructed:
        if char.isalpha() or char in {"^", "&", "\\", "/", "[", "]"}:
            if current:
                formatted.append(''.join(current))
                current = []
            formatted.append(char)
        else:
            current.append(char)

    if current:
        formatted.append(''.join(current))

    return formatted


class MockParam:
    def __init__(self, value, _type):
        self.value = value
        self._type = _type

def process_file_cluster(path ,tree, clustering_ratio=0.5):
    clusteredFolderPath = f"{tree}Clustered"
    picklePath = f"{tree}Pickle"
    os.makedirs(clusteredFolderPath, exist_ok=True)
    uniqueValues = set()
    clusterNumTotal = 0
    with open(path, "r") as f:
        sequence = "".join(line.strip() for line in f)
    path = os.path.basename(path)
    fileName = path.split(os.sep)[-1]
    params, mystring = parameterize(sequence)
    commands = set(["F", "-", "+", "^", "&", "\\", "/", "[", "]"])
    parser = Parser()
    nonCmd = ""
    for c in sequence:
        if c in commands:
            if nonCmd:
                parser.feed(nonCmd)
                nonCmd = ""
            parser.feed(c)
        elif c.isdigit() or c == '.':
            nonCmd += c
        else:
            if nonCmd:
                parser.feed(nonCmd)
                nonCmd = ""
            parser.feed(c)
    if nonCmd:
        parser.feed(nonCmd)

    grammar = Grammar(parser.tree)
    rulesFromGrammar = get_rules_grammar(grammar)

    distValue = []
    angleValue = []
    values = []

    for p in params:
        param = params[p]
        val = round(param.value, 2)
        if param._type == "D":
            distValue.append(val)
            values.append(val)
        elif param._type == "A":
            angleValue.append(val)
            values.append(val)

    paramTable = dict()  # param.value : id
    paramTableDist = dict()
    table = dict()
    for idx, p in enumerate(params):
        if params[p]._type == "A":
            key = float(params[p].value)
            key = round(key, 2)
            try:
                paramTable[key].add(idx)
                table[key].add(idx)
            except:
                paramTable[key] = set()
                paramTable[key].add(idx)
                table[key] = set()
                table[key].add(idx)

        if params[p]._type == "D":
            key = float(params[p].value)
            key = round(key, 2)
            try:
                paramTable[key].add(idx)
                table[key].add(idx)
            except:
                paramTable[key] = set()
                paramTable[key].add(idx)
                table[key] = set()
                table[key].add(idx)

    UNIQUE_ANGLES = len(set(angleValue))
    UNIQUE_DIST = len(set(distValue))
    NUM = int(len(set(values)) * clustering_ratio)
    clusterNumTotal += NUM
    cluster = KMeans(n_clusters=NUM)
    angleValue = np.asarray(angleValue)
    np_values = np.asarray(values)
    ids = cluster.fit_predict(np_values.reshape(-1, 1))
    KmeanValues = cluster.cluster_centers_
    clusterLabels = cluster.labels_
    b4, after = [], []
    for i in range(0, len(np_values)):
        angle = np_values[i]
        # get params id using angle
        ids = table[angle]
        # Get cluster id
        cluster_id = clusterLabels[i]
        # Get KmeanValue using cluster id
        clusterVal = round(KmeanValues[cluster_id][0], 2)
        uniqueValues.add(clusterVal)
        # Update the param value using the param id
        for id in ids:
            b4.append(params[id].value)
            params[id].value = clusterVal
            after.append(math.degrees(params[id].value))

    rulesFromGrammar = get_rules_grammar(grammar)
    firstRule = rulesFromGrammar[0]
    while not isConverted(firstRule):
        firstRule = convert(firstRule, rulesFromGrammar)
    parameterized = "".join(firstRule)

    reConstructed = reverseParameterClustering(parameterized, params)
    reConstructedFormatted = reFormatString(reConstructed)

    fileName = path.split(".")[0]
    newFileNameGenSeq = f"./{clusteredFolderPath}/{fileName}.lstring"
    with open(str(newFileNameGenSeq), "w") as the_file:
        for command in reConstructedFormatted:
            the_file.write(f"{command}\n")
    import pickle

    pickle_path = os.path.join(picklePath, fileName + ".pkl")
    os.makedirs(picklePath, exist_ok=True)
    with open(pickle_path, "wb") as fp:
        pickle.dump(uniqueValues, fp)
    return None

def number2letter(n, b=string.ascii_uppercase):
    d, m = divmod(n, len(b))
    return number2letter(d - 1, b) + b[m] if d else b[m]


def letter2number(number):
    col_number = (
            sum([(ord(number.upper()[-i - 1]) - 64) * 26 ** i for i in range(len(number))])
            - 1
    )
    return col_number


def get_table(path):
    with open(path, "rb") as f:
        table = pickle.load(f)

    alphabetTable = {val: number2letter(idx) for idx, val in enumerate(sorted(table))}

    print("Alphabet Table:")  # 打印生成的字母表
    for key, value in alphabetTable.items():
        print(f"{key}: {value}")

    return alphabetTable

import os

import os

import os


import os



def convertLetter(path, combined_pickle_path):
    # 确保转换后的路径目录存在
    convertedPath = os.path.join(os.path.dirname(path), "AcaciaConverted")
    os.makedirs(convertedPath, exist_ok=True)
    print(f"Converted path: {convertedPath}")

    # 读取并处理文件内容
    with open(path, "r") as f:
        sequence = "".join(line.strip() for line in f)
    print(f"Sequence: {sequence}")

    # 提取文件名
    fileName = os.path.basename(path)
    print(f"Original file name: {fileName}")

    # 参数化序列
    params, mystring = parameterize(sequence)
    print(f"Parameterize - params: {params}")
    print(f"Parameterize - mystring type: {type(mystring)}")
    print(f"Parameterize - mystring: {mystring}")

    # 提取mystring内容（如适用）
    if isinstance(mystring, dict) and len(mystring) == 1:
        mystring = next(iter(mystring.values()))[0]
    print(f"Extracted mystring: {mystring}")

    # 定义命令集合
    commands = set(["F", "-", "+", "^", "&", "\\", "/", "[", "]"])

    # 从pickle文件获取字母表映射表
    alphabetTable = get_table(combined_pickle_path)
    print(f"Alphabet Table: {alphabetTable}")

    # 替换参数值为字母
    for p in params:
        value = params[p].value
        if value in alphabetTable:
            letterValue = alphabetTable[value]
            params[p].value = letterValue
    print(f"Updated params: {params}")

    # 初始化解析器并处理字符串
    parser = Parser()
    nonCmd = ""

    for c in mystring:
        if isinstance(c, str):
            if c in commands:
                if nonCmd:
                    parser.feed(nonCmd)
                    nonCmd = ""
                parser.feed(c)
            elif c.isdigit() or c == '.':
                nonCmd += c
            else:
                if nonCmd:
                    parser.feed(nonCmd)
                    nonCmd = ""
                parser.feed(c)
        else:
            print(f"Unexpected type: {type(c)} for character {c}")

    if nonCmd:
        parser.feed(nonCmd)
    print(f"Parser tree: {parser.tree}")

    # 生成语法和规则
    grammar = Grammar(parser.tree)
    rulesFromGrammar = get_rules_grammar(grammar)
    print(f"Rules from Grammar: {rulesFromGrammar}")

    if rulesFromGrammar:
        firstRule = list(rulesFromGrammar.values())[0][0]
        print(f"First rule before conversion: {firstRule}")
        while not isConverted(firstRule):
            firstRule = convert(firstRule, [rule for sublist in rulesFromGrammar.values() for rule in sublist])
        print(f"First rule after conversion: {firstRule}")

        # 重构字符串
        parameterized = "".join(firstRule)
        reConstructed = reverseParameterClustering(parameterized, params)
        reConstructedFormatted = reFormatString(reConstructed)
        print(f"Reconstructed formatted string: {reConstructedFormatted}")

        # 生成新的文件路径并写入结果
        newFileNameGenSeq = os.path.join(convertedPath, f"{os.path.splitext(fileName)[0]}.lstring")
        print(f"New file name generated sequence path: {newFileNameGenSeq}")

        try:
            with open(newFileNameGenSeq, "w") as the_file:
                for command in reConstructedFormatted:
                    the_file.write(f"{command}\n")
            print(f"File created successfully: {newFileNameGenSeq}")
        except Exception as e:
            print(f"Error writing file: {e}")
    else:
        print(f"Error: No rules generated from grammar for file {fileName}")



def combine_all_pickles(tree, picklePath):
    pickles = glob(f"./{picklePath}/*")
    table = set()
    for p in pickles:
        with open(p, "rb") as f:
            p_val = pickle.load(f)
            print(f"Content of {p}: {p_val}")
            table |= p_val
    print(f"Combined table: {table}")  # 打印合并后的内容

    ratio_int = int(clustering_ratio * 100)
    combined_pickle_name = f"{tree}_{len(pickles)}_set_ratio{ratio_int}.pkl"
    with open(combined_pickle_name, "wb") as f:
        pickle.dump(table, f)
    return combined_pickle_name



def getPair(ls):
    _op, _cl = [], []
    brakcetList = []

    for idx, l in enumerate(ls):
        if l == "[":
            _op.append(idx)
            brakcetList.append(("[", idx))
        if l == "]":
            _cl.append(idx)
            brakcetList.append(("]", idx))
    stck = []
    pair = dict()
    for ele1, ele2 in brakcetList:
        if "[" in ele1:
            stck.append((ele1, ele2))
        elif "]" in ele1:
            s, e = stck.pop()[1], ele2
            pair[s] = e
    return pair


def isBracketFree(lstring):
    return "[" in lstring and "]" in lstring


def getGraveliusOrder(string, index):
    order = 0
    for s in string[: index + 1]:
        if s == "[":
            order += 1
        elif s == "]":
            order -= 1
    return order


def readLStrings(path):
    with open(path) as file:
        lines = file.readlines()
    strings = ""
    for l in lines:
        l = l.rstrip()
        strings += l
    return strings


def getMaxGraveliusOrder(string):
    order = 0
    maxOrder = -1
    for s in string:
        if s == "[":
            order += 1
            if maxOrder < order:
                maxOrder = order
        elif s == "]":
            order -= 1
    return maxOrder


def getRuleName(order, newRules):
    return f"R({order}@{len(newRules)})"





def get_rules(strings):
    return re.findall(r"R\(\w+@\w+\)", strings)


def get_Rxy(rule):
    x = rule[rule.index("(") + 1: rule.index("@")]
    y = rule[rule.index("@") + 1: rule.index(")")]
    return x, y


def get_geoInfo(strings):
    geoInfo = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for s in strings:
        if s in symbols:
            geoInfo += s
    return geoInfo


def get_geoInfoCount(strings):
    count = {"^": 0, "F": 0, "R": 0, "+": 0, "/": 0, "&": 0, "\\": 0, "-": 0}
    for key in count.keys():
        count[key] = strings.count(key)
    keys = sorted(list(count.keys()))
    res = " ".join(f"{key}{count[key]}" for key in keys).rstrip()
    return res


def getSubRuleToken(strings):
    return "".join(f"R({st})" for st in strings)


def getToken(tree, key, idx, rule, geoInfo):
    treeToken = tree[0].upper()
    start = f"<{treeToken}{key}@{idx} {geoInfo}>"
    if geoInfo == "":
        start = f"<{treeToken}{key}@{idx}>"
    newSeed = f"{start}{rule}<END>"
    return newSeed


def get_value(values):
    temp = ""
    symbols = set(["^", "F", "R", "&", "+", "-", "\\", "/"])
    for s in values:
        temp += s
        if s == ")":
            temp += " "
    return temp.rstrip()


def get_train_data(files):
    trainDataPath = f"{tree}TrainData"
    treeToken = tree[0].upper()
    os.makedirs(trainDataPath, exist_ok=True)

    for _file in tqdm(files):
        basename = os.path.basename(_file)
        trainData = defaultdict(list)

        with open(_file) as json_file:
            data = json.load(json_file)

        if not isinstance(data, dict):
            print(f"Warning: Expected a dictionary, got {type(data)} for file {_file}")
            continue

        keys = list(data.keys())
        keys.sort(key=int)

        valueLinkTable = {}
        valueTable = {}
        originTable = {}
        geoInfoTable = {}

        for order in data.keys():
            if not isinstance(data[order], dict):
                print(f"Warning: Expected a dictionary for order, got {type(data[order])} in file {_file}")
                continue

            for key in data[order].keys():
                rules = get_rules(data[order][key][0])
                geoInfoTable[f"R({order}@{key})"] = data[order][key][0]
                for r in rules:
                    x, y = get_Rxy(r)
                    originTable[f"R({x}@{y})"] = f"R({order}@{key})"

        for order in keys:
            orderValues = data[order]
            if not isinstance(orderValues, dict):
                print(f"Warning: Expected a dictionary for order values, got {type(orderValues)} in file {_file}")
                continue

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
                token = getToken(tree, order, idx, vals, spaceGeoInfo)
                trainData[order].append(token)

        for order in trainData.keys():
            temp = []
            for value in trainData[order]:
                temp_v = value.replace("R(", f"{treeToken}(")
                temp.append(temp_v)
            trainData[order] = temp

        with open(f"./{trainDataPath}/{basename}", "w") as outfile:
            json.dump(trainData, outfile, sort_keys=True, indent=4)


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


def split_to_train_test(files):
    csvFilePath = f"{tree}CSV"
    os.makedirs(csvFilePath, exist_ok=True)
    t_start_list, t_end_list = [], []
    for f in tqdm(files[:]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if order == "0":
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        df = pd.DataFrame(list(zip(starts_list, ends_list)), columns=["SRC", "TRG"])
        os.makedirs(f"{csvFilePath}", exist_ok=True)
        df.to_csv(f"./{csvFilePath}/{basename}.csv", index=False)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)
    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_total.csv", index=False)
    starts_counter = []
    s_counter = Counter()
    for start_list in tqdm(t_start_list):
        s = start_list.split()
        s_counter.update(s)
    del starts_counter
    gc.collect()
    ends_counter = []
    e_counter = Counter()
    for end_list in tqdm(t_end_list):
        e = end_list.split()
        e_counter.update(e)
    del ends_counter
    gc.collect()
    with gzip.open(f"./{tree}_S_counter_2.pkl", "wb") as f:
        pickle.dump(s_counter, f)
    with gzip.open(f"./{tree}_E_counter_2.pkl", "wb") as f:
        pickle.dump(e_counter, f)
    t_start_list, t_end_list = [], []
    for f in tqdm(files[:-100]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if order == "0":
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)
    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_train_total.csv", index=False)
    train_len = int(len(files) * 0.8)
    t_start_list, t_end_list = [], []
    for f in tqdm(files[-100:]):
        basename = os.path.basename(f).replace(".json", "")
        with open(f) as json_file:
            json_data = json.load(json_file)
        keys = json_data.keys()
        starts_list, ends_list = [], []
        for key in keys:
            orders = json_data[key]
            for order in orders:
                starts, ends = get_start_data_tokens(order)
                starts = starts.replace("<", "").replace(">", "")
                start_token = starts.split()[0]
                if order == "0":
                    starts = f"[{start_token}]"
                else:
                    parent = ' '.join(starts.split()[1:])
                    starts = f"[{start_token}] {parent}"
                starts_list.append(starts)
                ends_list.append(ends)
        t_start_list.extend(starts_list)
        t_end_list.extend(ends_list)
    df = pd.DataFrame(list(zip(t_start_list, t_end_list)), columns=["SRC", "TRG"])
    df.to_csv(f"./{csvFilePath}/{tree}_test_total.csv", index=False)

def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def process_file_wrapper(args):
    path, tree, clustering_ratio = args
    process_file_cluster(path, tree, clustering_ratio)

def get_values(path, tree):
    print(f"get_values() called with path: {path}")

    try:
        with open(path, 'r') as file:
            content = file.read()
            print(f"File content from {path[:50]}: {content[:100]}...")  # Print the first 100 characters for debugging

        # Update the paths to match the new structure
        ordersFolderPath = os.path.join(tree + "Orders")
        tablePath = os.path.join(tree + "Tables")
        os.makedirs(ordersFolderPath, exist_ok=True)
        os.makedirs(tablePath, exist_ok=True)

        # Simulate processing and creating orderValue (replace with actual logic)
        orderValue = defaultdict(list)
        orderValue["0"] = ["rule1"]
        orderValue["1"] = ["rule2"]

        name = os.path.basename(path).split(".")[0]
        order_json_path = os.path.join(ordersFolderPath, f"{tree}Order_{name}.json")
        table_json_path = os.path.join(tablePath, f"{tree}Table_{name}.json")

        print(f"Writing to {order_json_path}")
        print(f"Writing to {table_json_path}")

        # Writing the JSON files
        with open(order_json_path, "w") as outfile:
            json.dump(orderValue, outfile, sort_keys=False, indent=4)
        with open(table_json_path, "w") as outfile:
            json.dump(orderValue, outfile, sort_keys=False, indent=4)

        print(f"Generated {order_json_path} and {table_json_path}")
    except Exception as e:
        print(f"Error processing file {path}: {e}")

    return 1

import os
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool


# Define functions: process_file_cluster, combine_all_pickles, convertLetter, get_values, get_train_data, split_to_train_test

def start_process(t):
    print(f"Preprocessing starts with {t}")
    global tree
    tree = t
    start_time = time.time()

    # Setting the directory paths based on the tree type
    picklePath = f"{tree}Pickle"
    clusteredFolderPath = f"{tree}Clustered"
    convertedPath = f"{tree}Converted"
    orderPath = f"{tree}Orders"
    tablePath = f"{tree}Tables"
    csvFilePath = f"{tree}CSV"
    trainDataPath = f"{tree}TrainData"
    clustering_ratio = 0.1  # Set your clustering ratio here

    # Ensure directories exist
    ensure_directories_exist([picklePath, clusteredFolderPath, convertedPath, orderPath, tablePath, csvFilePath, trainDataPath])

    print(f"[1] Processing Clustering, path: {tree}")
    files = glob(os.path.join(tree, "*.lstring"))[:10]
    sorted_files = sorted(files)

    # Check if files are retrieved correctly
    print("Files to process for clustering:", sorted_files)

    p = mp.Pool(mp.cpu_count())
    for path in sorted_files:
        p.apply_async(process_file_cluster, args=(path, tree, clustering_ratio))
    p.close()
    p.join()

    with Pool(mp.cpu_count()) as p:
        p.map(process_file_wrapper, [(path, tree, clustering_ratio) for path in files])

    print(f"[1] Completed!, Processing Clustering, path: {t}")

    print("[2] Now, combining all pickle files")
    combined_pickle_path = combine_all_pickles(tree, picklePath)
    clusteredFiles = glob(os.path.join(clusteredFolderPath, "*.lstring"))
    print("Clustered files:", clusteredFiles)  # Debugging print
    for path in tqdm(clusteredFiles[:]):
        convertLetter(path, combined_pickle_path)
    print("[2] completed to convert all lstring files!")

    print("[3] Now, extracting lstring files to orders")
    files = glob(os.path.join(orderPath, "*.lstring"))  # Use orderPath here
    print("Files to process for extracting lstring files to orders:", files)  # Debugging print

    # Verify files exist before processing
    for path in files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
        else:
            print(f"Processing file: {path}")

    p = mp.Pool(mp.cpu_count())
    for path in files:
        print(f"Applying get_values to {path}")
        p.apply_async(get_values, args=(path, tree))
    p.close()
    p.join()
    print(tablePath)
    print("[3] Finished to extract the lstring information")
    #第一次生成之后记得暂时删除这段，否则影响后续
    allfiles = os.listdir(convertedPath)

    for f in allfiles:
        src_path = os.path.join(convertedPath, f)
        dst_path = os.path.join(orderPath, f)
        os.rename(src_path, dst_path)
    # Verify that the .json files were created
    generated_order_json_files = glob(os.path.join(orderPath, "*.json"))
    generated_table_json_files = glob(os.path.join(tablePath, "*.json"))
    print("Generated order JSON files:", generated_order_json_files)
    print("Generated table JSON files:", generated_table_json_files)

    if not generated_order_json_files:
        print("No order JSON files generated.")
    if not generated_table_json_files:
        print("No table JSON files generated.")
    print(tablePath)
    print("[4] Started generate csv files for the seq2seq")
    tableFiles = glob(os.path.join(tablePath, "*.json"))
    print("Table files to process for seq2seq:", tableFiles)  # Debugging print


    split_to_train_test(files)
    print(f"[4] Done! train and test csv files are at {csvFilePath}")

    seconds = time.time() - start_time
    print("Time Taken:", time.strftime("%H:%M:%S", time.gmtime(seconds)))


if __name__ == "__main__":
    tree_options = ["Acacia"]
    for t in tree_options:
        start_process(t)