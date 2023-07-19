import pickle
import faiss
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer as SBert
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QScrollArea

path_dic = {
    '純西南官話方言篇語料': ['index（西南官話）.pkl', 'combined_df（西南官話）.pkl'],
    '包括西南官話區少數民族語、風俗物產篇語料': ['index（包括其他）.pkl', 'combined_df（包括其他）.pkl']
} #根据自己实际情况设置语料索引和文本

def load_pickle_files(selected_corpus): #根据自己实际情况增减语料索引和文本的读取数目
    path_1, path_2 = path_dic[selected_corpus]
    with open(path_1, 'rb') as f:
        index = pickle.load(f)
    with open(path_2, 'rb') as f:
        combined_df = pickle.load(f)
    return index, combined_df

def search(query_entry, k_entry, output_textbox, corpus_var):
    """
    根据输入的查询文本，在索引中搜索相似的句子，并在输出框中显示结果。

    参数：
    - query_entry: 查询输入框的引用
    - k_entry: 查询条数输入框的引用
    - output_textbox: 输出文本框的引用
    - corpus_var: 下拉框变量的引用
    """
    query_text = query_entry.text()  # 获取输入的查询文本
    query_embedding = model.encode([query_text], device=device)

    k = int(k_entry.text())  # 获取输入的查询条数

    selected_corpus = corpus_var.currentText()
    index, combined_df = load_pickle_files(selected_corpus)

    _, top_k_indices = index.search(query_embedding, k)

    top_k_sentences = combined_df.loc[top_k_indices[0]]
    output_text = ''
    for _, row in top_k_sentences.iterrows():
        title = row['title']
        text = row['text']
        output_text += title + '\n' + text + '\n------\n'
    # 将搜索结果 output_text 显示在输出文本框中，同时在显示结果之前清空文本框的内容
    output_textbox.setPlainText(output_text)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('語義相似度查詢')
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        query_label = QLabel('檢索內容：')
        self.query_entry = QLineEdit()
        main_layout.addWidget(query_label)
        main_layout.addWidget(self.query_entry)

        k_label = QLabel('檢索條數：')
        self.k_entry = QLineEdit()
        main_layout.addWidget(k_label)
        main_layout.addWidget(self.k_entry)

        corpus_label = QLabel('請選擇語料：')
        self.corpus_dropdown = QComboBox()
        self.corpus_dropdown.addItems(path_dic.keys())
        main_layout.addWidget(corpus_label)
        main_layout.addWidget(self.corpus_dropdown)

        search_button = QPushButton('檢索')
        search_button.clicked.connect(self.perform_search)
        main_layout.addWidget(search_button)

        self.output_textbox = QTextEdit()
        self.output_textbox.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.output_textbox)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

    def perform_search(self):
        search(self.query_entry, self.k_entry, self.output_textbox, self.corpus_dropdown)


if __name__ == '__main__':
    app = QApplication([])
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SBert('./siku-bert').to(device)
    window = MainWindow()
    window.show()
    app.exec_()
