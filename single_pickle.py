import re
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os


# 提取参数的函数
def extract_parameters(sequence):
    pattern = r'\((\d+\.\d+)\)'
    matches = re.findall(pattern, sequence)
    return np.array([float(match) for match in matches])


# 聚类函数
def kmeans_clustering(values, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(values.reshape(-1, 1))
    return kmeans.labels_, kmeans.cluster_centers_


# 保存到pickle文件
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


# 转换函数
def convert_clusters_to_dict(labels, centers):
    cluster_dict = {}
    for idx, center in enumerate(centers.flatten()):
        cluster_dict[center] = chr(65 + idx)  # A, B, C, ...
    return cluster_dict


# 处理单个文件
def process_file(filepath, n_clusters, output_folder):
    with open(filepath, 'r') as file:
        sequence = file.read()

    # 提取参数
    values = extract_parameters(sequence)

    # 如果提取到的参数数量小于 n_clusters，返回空字典
    if len(values) < n_clusters:
        print(f"Not enough data to cluster in file {filepath}. Required: {n_clusters}, Found: {len(values)}")
        return {}

    # 聚类
    labels, centers = kmeans_clustering(values, n_clusters)

    # 转换聚类结果为字典
    cluster_dict = convert_clusters_to_dict(labels, centers)

    # 保存结果到pickle文件
    filename = os.path.basename(filepath)
    pickle_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_clusters.pkl")
    save_to_pickle(cluster_dict, pickle_filename)

    # 打印结果
    print(f"Processed file: {filepath}")
    print(f"Cluster dictionary: {cluster_dict}")


# 处理文件夹中的所有文件
def process_folder(input_folder, output_folder, n_clusters):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if os.path.isfile(filepath):
            process_file(filepath, n_clusters, output_folder)


# 示例使用
input_folder = "C:\\Users\\hp\\PycharmProjects\\ACM\\AcaciaClustered"  # 输入文件夹路径
output_folder = "C:\\Users\\hp\\PycharmProjects\\ACM\\AcaciaPickle"  # 输出文件夹路径
n_clusters = 4  # 示例值，根据需要调整
process_folder(input_folder, output_folder, n_clusters)


# 合并pickle文件并生成特定名称
import pickle
import os

def combine_pickles(pickle_folder, output_filename):
    combined_data = []

    # 遍历所有pickle文件并加载数据
    for filename in os.listdir(pickle_folder):
        if filename.endswith('.pkl'):
            filepath = os.path.join(pickle_folder, filename)
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                combined_data.append(data)

    # 保存合并后的数据到一个新的pickle文件
    with open(output_filename, 'wb') as output_file:
        pickle.dump(combined_data, output_file)

    print(f"Combined {len(combined_data)} pickle files into {output_filename}")

# 示例使用
pickle_folder = "C:\\Users\\hp\\PycharmProjects\\ACM\\AcaciaPickle"  # 替换为实际的pickle文件夹路径
output_filename = f"Acacia_{len(os.listdir(pickle_folder))}_set_ratio.pkl"  # 示例输出文件名，根据需要调整
combine_pickles(pickle_folder, output_filename)




