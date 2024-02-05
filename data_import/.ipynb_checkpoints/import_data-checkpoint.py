# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:20:23 2023

@author: mi
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import os
import tqdm
import openai


def to_embeddings(items):
    '''
    使用OpenAI的API将句子转换为向量,需要科学上网

    Parameters
    ----------
    items : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    sentence_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=items[0]
    )
    return [items[0], items[1], sentence_embeddings["data"][0]["embedding"]]

from qdrant_client import QdrantClient



if __name__ == '__main__':
    with open('key.txt','r') as f:
         key1,key2= f.readlines()
    
    client = QdrantClient(
        url="https://29d63520-276c-45b7-9d8e-d6bfa189a239.us-east-1-0.aws.cloud.qdrant.io:6333", 
        api_key=key2,
    )
    
    collection_name = "demo"
    openai.api_key=key1.replace('\n','')

    # 创建collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE), #向量参数
    )

    count = 0
    with open('./source_data/data.txt',encoding='gbk') as f:
        all_text = f.read()
        text_list = all_text.split('\n\n\n')
        for text in tqdm.tqdm(text_list):
            parts = text.split('\n')
            item = to_embeddings(parts)  #将数据向量化
            #将向量和文件名、文件内容一起作为一个文档插入到 Qdrant 数据库中
            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[
                    PointStruct(id=count, vector=item[2], payload={"question": item[0], "answer": item[1]}),
                ],
            )
            count += 1
