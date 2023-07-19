import os
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer as SBert
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import faiss

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SBert('SIKU-BERT/sikubert').to(device)

# 数据集和列的相关信息
data_folder = './input' 
csv_column = 'text'

# 加载数据集
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]
dataframes = []
for file in csv_files:
    csv_path = os.path.join(data_folder, file)
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as file:
        df = pd.read_csv(file)
    dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)


dataset = Dataset.from_pandas(combined_df)

# 创建数据加载器
batch_size = 5000

def collate_fn(batch):
    sentences = [item[csv_column] for item in batch]
    return {csv_column: sentences}

dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=80, shuffle=False, collate_fn=collate_fn)

# 获取句子嵌入并显示进度条
sentences_embeddings = []
with torch.no_grad(), tqdm(total=len(dataloader), desc='Processing') as pbar:
    for batch in dataloader:
        sentences = batch[csv_column]
        encoded_sentences = model.encode(sentences, convert_to_tensor=True, device=device)
        sentences_embeddings.append(encoded_sentences)

        pbar.update(1)

sentences_embeddings = torch.cat(sentences_embeddings, dim=0)

# 转换为NumPy数组
sentences_np = sentences_embeddings.cpu().numpy()

# 构建faiss索引
d = sentences_np.shape[1]  # 嵌入维度
index = faiss.IndexFlatIP(d)  # 构建内积相似度索引
index.add(sentences_np)  # 添加句子嵌入到索引中

# 保存起来以便未来使用
output_path_1 = './combined_df.pkl'  
with open(output_path_1, 'wb') as f:
    pickle.dump(combined_df, f)

output_path_2 = './index.pkl' 
with open(output_path_2, 'wb') as f:
    pickle.dump(index, f)