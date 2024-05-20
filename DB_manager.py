# https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search.py

import tiktoken
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import numpy as np
# import pandas as pd
import os
import datetime
import glob

import vibrato
import zstandard
import requests

import asyncio
import itertools
from pgvector.psycopg import register_vector_async
import psycopg
from sentence_transformers import CrossEncoder, SentenceTransformer


# DB parametr
#location = ":memory:"

MAX_TOKEN = 5000

def check_token(text):
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    token_integers = encoding.encode(text)
    num_tokens = len(token_integers)
    
    return num_tokens


class pg_db():
    def __init__(self, dbname, tablename, device) -> None:
        self.dbname = dbname
        self.tablename = tablename
        # model_name = 'intfloat/multilingual-e5-large'
        # model_name = 'pkshatech/GLuCoSE-base-ja'
        model_name = 'tohoku-nlp/bert-base-japanese'
        self.model = SentenceTransformer(model_name, device=device)
        self.model_dim = 768
        
        # 日本語ストップワード辞書
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt"
        self.stopwords_jp = requests.get(url).text.split("\n")
        
        #https://github.com/daac-tools/vibrato/releases/download/v0.5.0/ipadic-mecab-2_7_0.tar.xz
        dctx = zstandard.ZstdDecompressor()
        with open('ipadic-mecab-2_7_0/system.dic.zst', 'rb') as fp:
            with dctx.stream_reader(fp) as dict_reader:
                self.tokenizer = vibrato.Vibrato(dict_reader.read())

    
    async def create_schema(self, conn, metadata):
        metadata_key = ' text, '.join(list(metadata.keys())) + ' text'
        # metadata_key = f'_em vector({self.model_dim}), '.join(list(metadata.keys())) + f' vector({self.model_dim})'
        
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS pg_bigm')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS btree_gin')
        await register_vector_async(conn)

        await conn.execute(f'DROP TABLE IF EXISTS {self.tablename}')
        await conn.execute(f'CREATE TABLE {self.tablename} (id bigserial PRIMARY KEY, content text, embedding vector({self.model_dim}), {metadata_key})')
        await conn.execute(f"CREATE INDEX ON {self.tablename} USING GIN (content gin_bigm_ops)")


    async def insert_data(self, conn, sentences, metadata):
        embeddings = self.model.encode(sentences)
        # model = SentenceTransformer('pkshatech/GLuCoSE-base-ja')

        metadata_keys = ', '.join(list(metadata.keys()))
        metadata_s = ', '.join(['%s' for _ in metadata.keys()])
        sql = f'INSERT INTO {self.tablename} (content, embedding, {metadata_keys}) VALUES ' + ', '.join([f'(%s, %s, {metadata_s})' for _ in sentences])
        params = list(itertools.chain(*zip(sentences, embeddings, *[[v]*len(sentences) for v in metadata.values()])))
        await conn.execute(sql, params)


    async def semantic_search(self, conn, query):
        # model = SentenceTransformer('pkshatech/GLuCoSE-base-ja')
        embedding = self.model.encode(query)
        embedding_str = ",".join([str(i) for i in embedding])

        async with conn.cursor() as cur:
            # await cur.execute(f"SELECT id, content FROM documents ORDER BY embedding <=> [{embedding_str}] LIMIT 5")
            await cur.execute(f"SELECT id, content, file_name, 1 - (embedding <=> '[{embedding_str}]') AS cosine_similarity FROM {self.tablename} LIMIT 5")
            return await cur.fetchall()


    async def keyword_search(self, conn, query):
        query_lst = self.preprocess_jp(query)
        like_query = ' OR '.join([f"content LIKE '%{q}%'" for q in query_lst])
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT id, content, file_name FROM {self.tablename} WHERE {like_query} LIMIT 5")
            return await cur.fetchall()


    def rerank(self, query, results):
        # deduplicate
        results = set(itertools.chain(*results))

        # re-rank
        encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = encoder.predict([(query, item[1]) for item in results])
        return [v for _, v in sorted(zip(scores, results), reverse=True)]


    def preprocess_jp(self, text: str):
        """日本語テキストを前処理してトークンリストを返す"""

        # 改行コードの除去
        text = text.replace("\n", "")

        # vibrato
        word_list = []
        pos_list = ["名詞"]
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            tag_lst = token.feature().split(',')
            if any(tag_lst[0] == pos for pos in pos_list):
                if tag_lst[0] == "名詞":
                    if tag_lst[1] == "一般":
                        word_list.append(token.surface())
                #if tag_lst[0] == "動詞":
                #    if tag_lst[1] != "非自立" and tag_lst[1] != "接尾":
                #        word_list.append(token.surface())
        """指定した単語リストからストップワードを除去した結果を返す"""
        word_list = [word for word in word_list if word not in self.stopwords_jp]
        
        return word_list

    def create_sentence(self, file_path):
        with open(file_path, 'r') as f:
                data = f.readlines()
        texts = "".join(data)
            
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="。", # チャンクの区切り文字リスト
            chunk_size=5000,           # チャンクの最大文字数
            chunk_overlap=2000,         # チャンク間の重複する文字数
        )
        sentences = text_splitter.split_text(texts)
        
        return sentences
                
    async def register(self, file_list):
        
        metadata = {'file_name': None, 'time': datetime.datetime.now()}
        conn = await psycopg.AsyncConnection.connect(dbname=self.dbname, autocommit=True)
        await self.create_schema(conn, metadata)
            
        for file_path in file_list:
            basename = os.path.basename(file_path)
            file_name = os.path.splitext(basename)[0]
            metadata = {'file_name': file_name, 'time': datetime.datetime.now()}
            
            sentences = self.create_sentence(file_path)
            await self.insert_data(conn, sentences, metadata)

    async def hybrid_search_results(self, query):
        conn = await psycopg.AsyncConnection.connect(dbname=self.dbname, autocommit=True)
        # perform queries in parallel
        results = await asyncio.gather(self.semantic_search(conn, query), self.keyword_search(conn, query))
        results = self.rerank(query, results)
        return results
        
    async def semantic_search_results(self, query):
        conn = await psycopg.AsyncConnection.connect(dbname=self.dbname, autocommit=True)
        results = await self.semantic_search(conn, query)
        return results
    
    async def keyword_search_results(self, query):
        conn = await psycopg.AsyncConnection.connect(dbname=self.dbname, autocommit=True)
        results = await self.keyword_search(conn, query)
        return results


    

if __name__ == '__main__': 
    table_name = 'tbl_GLu'
    dir_path = f'../data/{table_name}/*' 
    
    pg_db_cls = pg_db(dbname='vector_db', tablename=table_name, device='cuda:0')
    
    file_list = glob.glob(dir_path)
    
    
    #if os.path.isdir(db_path):
    #    shutil.rmtree(db_path)
    
    asyncio.run(pg_db_cls.register(file_list))
    # asyncio.run(pg_db_cls.semantic_search_results('流量観測について知りたい'))
        