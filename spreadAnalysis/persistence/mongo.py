# MONGODB COLLECTIONS

"""
platform
pull
query
actor
url
post
actor_website_url
actor_url
query_url
url_post
actor_post
query_post

"""

# MONGODB MANDATORY FIELDS

"""
ALL COLLECTIONS:

- _id
- insert_time
- update_time

"""

"""
platform

- platform_name
- platform_api_url
- available

pull

- input
- input_type
- method
- last_run
- last_run_issue
- attempts ~isList
    - issue

queries

- query
- input_file
- title

actors

- actor
- input_file
- title

urls

- url
- input_file
- title

posts

- message_id

"""

import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from spreadAnalysis.utils.link_utils import LinkCleaner
from pymongo import InsertOne, DeleteOne, ReplaceOne, UpdateOne
from spreadAnalysis.persistence.schemas import Spread


class MongoDatabase:

    def __init__(self):

        self.client = MongoClient('mongodb://localhost:27017/')
        self.database = self.client["spreadAnalysis"]

    def unpack_docs(self,cursor):

        return [doc for doc in cursor]

    def get_ids_from_db(self,db,use_col=None):

        ids = {d[use_col]:d["_id"] for d in db.find({},{ "_id": 1, use_col: 1})}
        return ids

    def get_keys_from_db(self,db,use_col=None):

        ks = set([d[use_col] for d in db.find({},{use_col: 1})])
        return ks

    def get_keys_from_nested(self,db,top_col,nest_col):

        all_keys = set([])
        comb_key = f"{top_col}.{nest_col}"
        for d in db.find({},{comb_key: 1}):
            for dd in d[top_col]:
                all_keys.add(dd[nest_col])

        ks = set([d[use_col] for d in db.find({},{use_col: 1})])
        return ks

    def get_key_pairs_from_db(self,db,col1,col2):

        kps = {}
        for d in db.find({},{col1: 1, col2: 1}):
            if not d[col1] in kps: kps[d[col1]]=set([])
            kps[d[col1]].add(d[col2])
        return kps

    def get_data_from_db(self,db):

        data = self.unpack_docs(db.find({}))
        return data

    def delete_with_key_val(self,db,key,val):

        if isinstance(key,tuple):
            db.delete_one({key[0]:val[0],key[1]:val[1]})
        else:
            db.delete_one({key:val})

    def test_update(self):

        db = self.database["post"]
        #db.bulk_write([InsertOne({'_id': 111, "foo":"bar"})])
        db.bulk_write([UpdateOne({'_id': 111}, {'$set': {'foo': 'sup'}})])

    def update_many(self,db,docs,key_col):

        bulks = []
        if len(docs) > 0:
            for doc in docs:
                doc["updated_at"]=datetime.now()
                if isinstance(key_col,tuple):
                    bulks.append(UpdateOne({key_col[0]:doc[key_col[0]],key_col[1]: doc[key_col[1]]},
                                {'$set': doc}))
                else:
                    bulks.append(UpdateOne({key_col:doc[key_col]},
                                {'$set': doc}))
            db.bulk_write(bulks)

    def insert_many(self,db,docs):

        bulks = []
        if len(docs) > 0:
            for doc in docs:
                doc["inserted_at"]=datetime.now()
                bulks.append(InsertOne(doc))
            db.bulk_write(bulks)

    def insert_one(self,db,doc):

        doc["inserted_at"]=datetime.now()
        db.insert_one(doc)

    def write_many(self,db,docs,key_col):

        bulks = []
        seen_ids = set([])
        if len(docs) > 0:
            if isinstance(key_col,tuple):
                for doc in docs:
                    if (doc[key_col[0]],doc[key_col[1]]) in seen_ids: continue
                    if db.count_documents({ key_col[0]: doc[key_col[0]],key_col[1]: doc[key_col[1]] }, limit = 1) != 0:
                        doc["updated_at"]=datetime.now()
                        bulks.append(UpdateOne({key_col[0]:doc[key_col[0]],key_col[1]: doc[key_col[1]]},
                                    {'$set': doc}))
                    else:
                        doc["inserted_at"]=datetime.now()
                        bulks.append(InsertOne(doc))
                    seen_ids.add((doc[key_col[0]],doc[key_col[1]]))
            else:
                for doc in docs:
                    if doc[key_col] in seen_ids: continue
                    if db.count_documents({ key_col: doc[key_col] }, limit = 1) != 0:
                        doc["updated_at"]=datetime.now()
                        bulks.append(UpdateOne({key_col:doc[key_col] },
                                    {'$set': doc}))
                    else:
                        doc["inserted_at"]=datetime.now()
                        bulks.append(InsertOne(doc))
                    seen_ids.add(doc[key_col])
            db.bulk_write(bulks)

    def insert_into_nested(self,db,doc,nest_keys):

        doc["updated_at"]=datetime.now()
        if isinstance(nest_keys[0],tuple):
            db.update(
                { nest_keys[0][0]: nest_keys[0][1], nest_keys[1][0]: nest_keys[1][1]},
                { "$push": {"attempts":doc}})
        else:
            db.update(
                { nest_keys[0]: nest_keys[1]},
                { "$push": {"attempts":doc}})


class MongoSpread(MongoDatabase):

    custom_file_schema = {  "actor":("Actors.xlsx","Actor"),
                            "query":("Queries.xlsx","Query"),
                            "url":("Urls.xlsx","Url")  }
    del custom_file_schema["query"]

    avlb_platforms = {"Facebook Page":"crowdtangle",
                        "Facebook Group":"crowdtangle",
                        "Twitter":"twitter2",
                        "Instagram":"crowdtangle",
                        "Reddit":"reddit",
                        "Youtube":"youtube",
                        "Tiktok":"tiktok",
                        "Vkontakte":"vkontakte",
                        "CrowdtangleApp":"crowdtangle_app",
                        "Google":"google",
                        "Majestic":"majestic",
                        "Telegram":"telegram",
                        "Gab":"gab"}

    avlb_endpoints = {  "Facebook Page":["actor","url","domain","query"],
                        "Facebook Group":["actor","url","domain","query"],
                        "Twitter":["actor","url","domain","query"],
                        "Instagram":["actor","url","domain","query"],
                        "Reddit":["url","domain","query"],
                        "Youtube":["actor"],
                        "Tiktok":["actor"],
                        "Vkontakte":["actor","url","domain","query"],
                        "CrowdtangleApp":["url"],
                        "Google":["url","query"],
                        "Majestic":["url","domain"],
                        "Telegram":["actor"],
                        "Gab":["actor"] }

    table_key_cols = {  "post":"message_id",
                        "pull":"input",
                        "url":"Url",
                        "actor":"Actor",
                        "url_post":("input","message_id"),
                        "actor_post":("input","message_id"),
                        "domain_url":("input","url")  }

    def __init__(self,custom_file_schema=None):
        MongoDatabase.__init__(self)

        if custom_file_schema is not None:
            self.custom_file_schema = custom_file_schema
        self.create_indexes()

    def create_indexes(self):

        for table,key_col in self.table_key_cols.items():
            if isinstance(key_col,str):
                self.database[table].create_index([ (key_col, -1) ])
            elif isinstance(key_col,tuple):
                self.database[table].create_index([ (key_col[0], -1), (key_col[1], -1)])

    def get_custom_file_as_df(self,file_path):

        df = pd.read_excel(file_path)
        df = df.where(pd.notnull(df), None)
        df = df.replace({np.nan: None})
        return df

    def update_custom_data(self,custom_path,custom_title=None,with_del=True,files_key_cols=None):

        if files_key_cols is None:
            files_key_cols = self.custom_file_schema

        if custom_title is None:
            custom_title = custom_path.split("/")[-1]

        for database, files in files_key_cols.items():
            db = self.database[database]
            file_path = files[0]
            key_col = files[1]
            prev_ids = self.get_ids_from_db(db,key_col)
            prev_data = self.get_data_from_db(db)
            actor_df = self.get_custom_file_as_df(custom_path+"/"+file_path)
            actor_df["org_project_title"]=custom_title
            actor_df["org_project_path"]=custom_path
            cols = actor_df.columns
            actor_docs = [{col:row[col] for col in cols} for i,row in actor_df.iterrows()]

            inserts, updates = [d for d in actor_docs if d[key_col] not in prev_ids],\
                [d for d in actor_docs if d[key_col] in prev_ids]
            self.insert_many(db,inserts)
            self.update_many(db,updates,key_col)

            if with_del:
                new_actor_keys = set([d[key_col] for d in actor_docs])
                for k,_id in prev_ids.items():
                    if k not in new_actor_keys:
                        self.delete_with_key_val(db,(key_col,"org_project_title"),(k,custom_title))

    def update_platform_info(self,with_test=False):

        db = self.database["platform"]
        db.drop()
        prev_ids = self.get_ids_from_db(db,"platform_dest")
        not_working_platforms = set(["CrowdtangleApp","Google"])

        inserts, updates = [], []
        for platform_dest, source in self.avlb_platforms.items():
            platform_doc = {}
            platform_doc["working"]=True
            if platform_dest in not_working_platforms: platform_doc["working"]=False
            platform_doc["platform_dest"]=platform_dest
            platform_doc["platform_source"]=source
            for endp in ["url","actor","domain","query"]:
                if endp in self.avlb_endpoints[platform_dest]:
                    platform_doc[endp+"_endpoint"]=True
                else:
                    platform_doc[endp+"_endpoint"]=False
            if platform_dest not in prev_ids:
                inserts.append(platform_doc)

        self.insert_many(db,inserts)

    def update_aliases(self,custom_path):

        prev_aliases = self.get_key_pairs_from_db(self.database["alias"],"actor","platform")
        actor_info = self.get_data_from_db(self.database["actor"])

        inserts, updates = [], []
        for doc in actor_info:
            for platform in list(self.avlb_platforms.keys()):
                if platform in doc and doc[platform] is not None:
                    alias = LinkCleaner().extract_username(doc[platform])
                    if alias is not None and len(alias) > 1:
                        alias_doc = {"actor":doc["Actor"],"platform":platform,"alias":alias}
                        if doc["Actor"] in prev_aliases and platform in prev_aliases[doc["Actor"]]:
                            updates.append(alias_doc)
                        else:
                            inserts.append(alias_doc)
        self.insert_many(self.database["alias"],inserts)
        self.update_many(self.database["alias"],updates,("actor","platform"))

    def update_bi_network(self,edge_key="message_id"):

        db = self.database["bi_network"]
        prev_edge_keys = self.get_keys_from_nested(db,"edges",edge_key)
        prev_node_pairs = self.get_key_pairs_from_db(self.database["bi_network"],"url","actor")
        actor_info = self.get_data_from_db(self.database["actor"])
        actor_info = {doc["Actor"]:doc for doc in actor_info}
        aliases = self.get_data_from_db(self.database["alias"])
        aliases = {doc["alias"]:doc for doc in aliases}
        #prev_actor_ids_post_ids = self.mdb.get_key_pairs_from_db(self.mdb.database["actor_post"],"input","message_id")
        prev_url_ids_post_ids = self.get_key_pairs_from_db(self.database["url_post"],"input","message_id")
        for url,posts in prev_url_ids_post_ids.items():
            for mid in posts:
                if not mid in prev_edge_keys:
                    post_data = self.database["post"].find({"message_id":mid})
                    url_node = url
                    url_label = url[:30]+"..."
                    actor_node = Spread_._get_actor_username(data=post_data,method=post_data["method"])
                    if actor_node in aliases:
                        actor_label = aliases[actor_node]+" ({0})".format(Spread_._get_platform(data=post_data,method=post_data["method"]))
                    else:
                        actor_label = Spread_._get_actor_name(data=post_data,method=post_data["method"])+" ({0})".format(Spread_._get_platform(data=post_data,method=post_data["method"]))
                    new_edge = None
