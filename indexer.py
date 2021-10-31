# import os
#
# from jina import DocumentArray, Executor, requests, Document
#
#
# import psycopg2  # 先安装模块：pip install psycopg2
# import numpy as np
#
#
# def str_to_array(string):
#     string_=string.split(']')[0][1:]
#     return np.fromstring(string_,dtype=float,sep=',')
#
# class AnnIndexer(Executor):
#     def __init__(self, task: str = 'search', *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # self.my_indexer = AnnoyIndex(512, 'euclidean')  #这里是distance
#         # self.task = task  # 加 定义了一个变量啥都不是
#         # if os.path.exists('landmark.ann') and self.task == 'search':
#         #     self.my_indexer.load('landmark.ann')  # super fast, will just mmap the file
#
#         self.conn = psycopg2.connect(database="", user="", password="", host="",
#                                 port="")
#         self.conn.autocommit = True
#
#
#     @requests(on='/index')
#     def index(self, docs: DocumentArray, **kwargs):
#         for doc in docs:
#             string = str(list(doc.blob)).split(']')[1]
#             print(len(list(string)))
#             # 获得游标对象
#             cursor = self.conn.cursor()
#             # sql语句，用于查询数据库的版本号
#             insert = "INSERT INTO t_training_global_feature VALUES ({0}, {1}, '{2}', {3})".format(int(doc.id), int(doc.text), list(doc.embedding), doc.uri)
#             # 执行语句
#             cursor.execute(insert)
#             cursor.close()
#
#             # self.my_indexer.add_item(int(doc.id), doc.embedding)
#
#     def close(self):  # 错误的地方
#         # if self.task == 'index':
#         #     # save ann
#         #     self.my_indexer.build(10)  # 10 trees
#         #     self.my_indexer.save('landmark.ann')
#         # super(AnnIndexer, self).close()
#         self.conn.close()
#
#
#     @requests(on='/search')
#     def search(self, docs: DocumentArray, parameters={}, **kwargs):
#         top_k = parameters.get('top_k', 20)
#         for doc in docs:
#             indexes, distances = self.my_indexer.get_nns_by_vector(doc.embedding, top_k, include_distances=True)  #这里是不是写反了 应该是query 找dataset
#             for idx, distance in zip(indexes, distances):
#                 print(idx, '->', distance)
#                 d = Document(id=idx)
#                 d.scores['distance'] = distance
#                 doc.matches.append(d)
#
#
