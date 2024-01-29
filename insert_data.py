from langchain import LLMChain
from langchain.document_loaders import DirectoryLoader,PyPDFLoader,JSONLoader,TextLoader
from langchain.document_loaders import Docx2txtLoader,UnstructuredHTMLLoader,CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
import re
import glob
import os

def cut_sent(para):
    '''
    分句函数
    '''
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return len(para.split("\n"))


def data_loader(file):
    '''
    按文件类型加载文件
    '''
    loader = None
    if os.path.splitext(file)[-1]=='.pdf':
        loader = PyPDFLoader(file)
    elif os.path.splitext(file)[-1] in ['.oc','.docx']:
        loader = Docx2txtLoader(file)
    elif os.path.splitext(file)[-1]=='.html':
         loader = UnstructuredHTMLLoader(file)
    elif os.path.splitext(file)[-1]=='.json':
         loader = JSONLoader(file)
    elif os.path.splitext(file)[-1]=='.csv':
         loader = CSVLoader(file)
    elif os.path.splitext(file)[-1] in ['.txt','.data','.dat']:
         loader = TextLoader(file)  
    return loader


def data_import(data_path):
    # 加载文件夹中的所有txt类型的文件
    file_list = glob.glob(os.path.join(data_path,'*'))

    # 初始化加载器
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 5,
        chunk_overlap  = 1,
        length_function = cut_sent,
        )
    
    ## 加载向量化模型
    embeddings = HuggingFaceEmbeddings(model_name='/root/ChatBot/QA/text2vec_cmed')
    
    for file in file_list:
        try:
            loader = data_loader(file)
            documents = loader.load()
        except:
            print('文件%s读取失败' % file)
            continue
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)

        # 将 document 通过 embeddings 对象计算 embedding 向量信息并临时存入 Qdrant 向量数据库，用于后续匹配查询
        vqdrant = Qdrant.from_documents(
        split_docs, embeddings, 
        path="./tmp/local_qdrant",
        collection_name="my_documents")


if __name__ == '__main__':
    data_path = 'data'
    data_import(data_path)
