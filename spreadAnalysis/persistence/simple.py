import pickle
import os
import sys
import pandas as pd

class Project:

    def __init__(self,folder_path,actor_meta_file=None,url_meta_file=None,init=False):

        self.main_path = folder_path
        self.init = init
        self.actor_meta_file = actor_meta_file
        self.url_meta_file = url_meta_file

    def get_project(self,format="python"):

        actor_data_path = f'{self.main_path}/actor_data.p'
        referal_data_path = f'{self.main_path}/referal_data.p'
        cleaned_urls_path = f'{self.main_path}/cleaned_urls.p'
        alias_data_path = f'{self.main_path}/alias_data.p'
        domain_data_path = f'{self.main_path}/domain_data.p'
        query_data_path = f'{self.main_path}/query_data.p'

        if self.actor_meta_file:
            if self.actor_meta_file == "special":
                supposed_filename = str(self.main_path).split("/data_collection")[0].\
                    split("/")[-1]+".xlsx"
                actor_meta_data_path = f'{self.main_path}/Start List {supposed_filename}'
            else:
                actor_meta_data_path = f'{self.main_path}/{self.actor_meta_file}'
        else:
            supposed_filename="Actors.xlsx"
            actor_meta_data_path = f'{self.main_path}/{supposed_filename}'
        actor_meta_data = pd.read_excel(actor_meta_data_path,engine="openpyxl")
        actor_meta_data = actor_meta_data.where(pd.notnull(actor_meta_data), None)

        if self.url_meta_file:
            url_meta_data_path = f'{self.main_path}/{self.url_meta_file}'
        else:
            supposed_filename="Urls.xlsx"
            url_meta_data_path = f'{self.main_path}/{supposed_filename}'
        url_meta_data = pd.read_excel(url_meta_data_path,engine="openpyxl")
        url_meta_data = url_meta_data.where(pd.notnull(url_meta_data), None)

        actor_data_backup_path = actor_data_path.replace(".p","")+"_backup.p"
        actor_data_backup_stable_path = actor_data_path.replace(".p","")+"_backup_STABLE.p"
        referal_data_backup_path = referal_data_path.replace(".p","")+"_backup.p"
        referal_data_backup_stable_path = referal_data_path.replace(".p","")+"_backup_STABLE.p"

        if os.path.isdir(self.main_path):

            actor_data_backup = {}
            referal_data_backup = {}
            actor_data = {}
            actor_data = FlatFile(actor_data_path,auto_init=self.init)
            referal_data = FlatFile(referal_data_path,auto_init=self.init)
            cleaned_urls = FlatFile(cleaned_urls_path,auto_init=self.init)
            alias_data = FlatFile(alias_data_path,auto_init=self.init)
            domain_data = FlatFile(domain_data_path,auto_init=self.init)
            query_data = FlatFile(query_data_path,auto_init=self.init)

            #actor_data_backup = FlatFile(actor_data_backup_path,auto_init=self.init)
            #actor_data_backup_stable = FlatFile(actor_data_backup_stable_path,auto_init=self.init)
            #referal_data_backup = FlatFile(referal_data_backup_path,auto_init=self.init)
            #referal_data_backup_stable = FlatFile(referal_data_backup_stable_path,auto_init=self.init)

            #actor_data_backup_stable.data = actor_data.data
            #referal_data_backup_stable.data = referal_data.data
            #actor_data_backup_stable.simple_update({})
            #referal_data_backup_stable.simple_update({})

            if format == "python":
                return actor_data, referal_data, cleaned_urls, alias_data, actor_meta_data,\
                    actor_data_backup, referal_data_backup
            if format == "dict":
                all_data = {"actor_data":actor_data,
                            "referal_data":referal_data,
                            "cleaned_urls":cleaned_urls,
                            "alias_data":alias_data,
                            "domain_data":domain_data,
                            "query_data":query_data,
                            "actor_meta_data":actor_meta_data,
                            "url_meta_data":url_meta_data,
                            "actor_data_backup":actor_data_backup,
                            "referal_data_backup":referal_data_backup}
                return all_data

        else:
            print ("Path is not a directory")
            sys.exit()

class FlatFile:

    def __init__(self,file_path,auto_init=True):

        self.file_path = file_path
        self.data = {}

        self.init_data(auto_init=auto_init)

    def init_data(self,auto_init=True):

        if not auto_init:
            if not os.path.isfile(self.file_path):
                print ("Cannot load {0} because files have not been initialized".\
                    format(self.file_path))
                sys.exit()

        if os.path.exists(self.file_path):
            self.data = pickle.load(open(self.file_path,"rb"))
        else:
            self.data = {}

    def simple_update(self,new_data):

        self.data.update(new_data)
        pickle.dump(self.data,open(self.file_path,"wb"))

    def full_update(self,new_data):

        if isinstance(self.data,list):
            prev_inputs = set([doc["input"] for doc in self.data])
            for doc in new_data:
                if doc["input"] not in prev_inputs:
                    self.data.append(doc)
                    prev_inputs.add(doc["input"])
            pickle.dump(self.data,open(self.file_path,"wb"))
        else:
            self.simple_update(new_data)
