**功能介绍**：
```
利用SBert和Faiss为语料编制索引，以实现语义相似度检索。
注意：本仓库只提供代码，语料需要根据自己的研究需求自备。
最终形成的软件界面如下：
```
![软件截图](Markdown_md_files/c81fb9e0-255e-11ee-b4f6-3bcae21a86d4.jpeg?v=1&type=image)

**代码说明**：
> 首先利用`构建faiss索引.py`构建索引:
> >编制索引时需要有GPU。

> >语料要求：在代码所在文件夹中创建`input`文件夹，放置utf-8编码的csv文件（可放置多个），文件分为两列，分别是`title`和`text`。其中`title`是语料来源，`text`是语料。尽量每行只放一句语料，且不能有空行。

> >语言模型：根据自己的语料情况设置语言模型。由于笔者使用的是古籍语料，因而选择利用四库全书语料预训练的siku-bert模型。

>> 输出内容：输出pickle文件，包括语料内容`combined_df.pkl`和索引`index.pkl`。如果需要分不同的语料库进行索引，需要分批次运行，并对输出结果重命名。例如笔者分两批运行，并重命名为`combined_df（西南官話）.pkl`、`index（西南官話）.pkl`、
`combined_df（包括其他）`、`index（包括其他）.pkl`。

> 其次利用`语义相似度检索.py`设置检索方式和GUI界面：
> > 此部分可以不用GPU。
 
> >根据输出的语料库数量修改pickle文件读取代码，具体修改处见代码中的注释。

>>由于笔者语料是繁体的古代地方志语料，为与语料用字统一，GUI界面的提示用字均为繁体。大家可根据实际情况改成简体，或设置不同的提示用字。

> 最后打包成exe文件：
> > 如果不需要将软件分享出去，可以不用打包。每次使用时运行`语义相似度检索.py`即可。

> > 建议在虚拟环境中打包，以减少exe文件的空间占用量。

> >打包时需要把代码、pickle文件、语言模型和图标文件放在同一个文件夹。

>>利用pyinstaller打包的命令如下，需根据自己实际情况设置图标、pickle文件、语言模型、代码的地址：
>`pyinstaller --noconfirm --onedir --console --icon "F:/語義相似度查詢（西南官話語料）/venv/圖標.ico" --add-data "F:/語義相似度查詢（西南官話語料）/venv/combined_df（包括其他）.pkl;." --add-data "F:/語義相似度查詢（西南官話語料）/venv/combined_df（西南官話）.pkl;." --add-data "F:/語義相似度查詢（西南官話語料）/venv/index（包括其他）.pkl;." --add-data "F:/語義相似度查詢（西南官話語料）/venv/index（西南官話）.pkl;." --add-data "F:/語義相似度查詢（西南官話語料）/venv/siku-bert;siku-bert/" --paths "D:/python/Lib/site-packages" --hidden-import "sklearn.neighbors.typedefs" --copy-metadata "tqdm" --copy-metadata "regex" --copy-metadata "requests" --copy-metadata "packaging" --copy-metadata "filelock" --copy-metadata "numpy" --copy-metadata "tokenizers" "F:/語義相似度查詢（西南官話語料）/venv/語義相似度檢索.py"`
