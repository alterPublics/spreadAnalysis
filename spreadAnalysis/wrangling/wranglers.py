from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis import _gvars as gvar
import csv
import pandas as pd
import random
import numpy as np

def insert_fourchan_data(path_to_file):

    cols = ["num", "subnum", "thread_num", "op",
            "timestamp", "timestamp_expired",
            "preview_orig", "preview_w", "preview_h",
            "media_filename", "media_w", "media_h",
            "media_size", "media_hash", "media_orig",
            "spoiler", "deleted", "capcode", "email",
            "name", "trip", "title", "comment", "sticky",
            "locked", "poster_hash", "poster_country", "exif"]

    row_count = 0
    n_cols = len(cols)
    with open(path_to_file,"r") as file_obj:
        reader_obj = csv.reader(file_obj, delimiter=',', quotechar='"')
        for line in reader_obj:
            #row_count+=1
            if len(line) == n_cols:
                doc = {cols[r]:line[r] for r in range(n_cols)}
            elif len(line) >= 23 and len(line) < n_cols:
                doc = {cols[r]:line[r] for r in range(len(line))}
            else:
                continue
            url = Spread._get_message_link(data=doc,method="fourchan")
            if url is not None:
                print (doc)
                row_count+=1
            if row_count > 10:

                sys.exit()

def delete_actor_source(main_path,actor=None,source=None,zero_hits=True):

    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    if actor is not None and source is not None:
        del actor_data.data[actor]["data"][source]

    if actor is None and source is not None:
        for actor in list(actor_data.data.keys()):
            if source in actor_data.data[actor]["data"]:
                if zero_hits:
                    if len(actor_data.data[actor]["data"][source]["output"]) < 1:
                        del actor_data.data[actor]["data"][source]
                else:
                    del actor_data.data[actor]["data"][source]

    #sys.exit()
    actor_data.simple_update({})

# Creates a list of aliases that matches the usernames of actors in the Actor Data Collection
def update_actor_aliases(main_path):

    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    alias_data = FlatFile(main_path+"/{0}".format("alias_data.p"))
    alias_data.data = {}
    for actor,dat in actor_data.data.items():
        dat = dat["meta"]
        for platform in list(gvar.SOURCES.keys()):
            if platform in dat and dat[platform] is not None:
                username = LinkCleaner().extract_username(dat[platform])
                if username == "s" or len(str(username)) < 2:
                    print (dat[platform])
                    #sys.exit()
                if username is not None:
                    alias_data.data[username]=actor
        if dat["Website"] is not None:
            domain = LinkCleaner().\
                extract_domain(dat["Website"])
            alias_data.data[domain]=actor
        alias_data.data[str(actor).lower()]=actor
    alias_data.simple_update({})

# Updates an Actor Data Collection by assigning each actor a list of URLs they shared.
# Also assigning all links from the Url Data Collection linking to an actors website.
def connect_actors_to_urls(main_path):

    referal_data = FlatFile(main_path+"/{0}".format("referal_data.p"))
    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    alias_data = FlatFile(main_path+"/{0}".format("alias_data.p"))
    domain_to_actor = {dat["meta"]["Website"]:actor for actor,dat in actor_data.data.items() if dat["meta"]["Website"] is not None}
    links_to_own_website = {actor:[] for actor,dat in actor_data.data.items()}
    links_shared_by_actor = {actor:[] for actor,dat in actor_data.data.items()}
    for url,url_dat in list(referal_data.data.items()):
        if "data" in url_dat: url_dat = url_dat["data"]
        for domain in list(domain_to_actor.keys()):
            if domain is not None and url is not None:
                if LinkCleaner().extract_domain(str(domain)) is not None and LinkCleaner().extract_domain(str(url)) is not None and LinkCleaner().extract_domain(str(domain)) in LinkCleaner().extract_domain(str(url)):
                    links_to_own_website[domain_to_actor[domain]].append(url)
        if "total_referals" in url_dat: del url_dat["total_referals"]
        for source in list(url_dat.keys()):
            if "output" in url_dat[source]:
                for post in url_dat[source]["output"]:
                    username = Spread._get_actor_username(method=url_dat[source]["method"],data=post)
                    actor_id = Spread._get_actor_id(method=url_dat[source]["method"],data=post)
                    actor_name = Spread._get_actor_name(method=url_dat[source]["method"],data=post).lower()
                    post_link = Spread._get_message_link(method=url_dat[source]["method"],data=post)
                    if username in alias_data.data and alias_data.data[username] in links_shared_by_actor:
                        links_shared_by_actor[alias_data.data[username]].append(url)
                        if post_link is not None: links_shared_by_actor[alias_data.data[username]].append(post_link)
                    elif actor_id in alias_data.data and alias_data.data[actor_id] in links_shared_by_actor:
                        links_shared_by_actor[alias_data.data[actor_id]].append(url)
                        if post_link is not None: links_shared_by_actor[alias_data.data[actor_id]].append(post_link)
                    elif actor_name in alias_data.data and alias_data.data[actor_name] in links_shared_by_actor:
                        links_shared_by_actor[alias_data.data[actor_name]].append(url)
                        if post_link is not None: links_shared_by_actor[alias_data.data[actor_name]].append(post_link)
    for actor,links in links_to_own_website.items():
        actor_data.data[actor]["links_to_own_website"]=list(set(links_to_own_website[actor]))
    for actor,links in links_shared_by_actor.items():
        actor_data.data[actor]["links_shared_by_actor"]=list(set(links_shared_by_actor[actor]))
    actor_data.simple_update({})

# Updates all meta data from Actor Data Collection based on an excel sheet containing meta data columns
def update_actor_meta(actor_list,actor_data,key_col="Actor",with_full_delete=False,delete_except=[]):

    cols = actor_list.columns
    actor_list = actor_list.where(pd.notnull(actor_list), None)
    actor_list = actor_list.replace({np.nan: None})
    new_actors = set([a for a in actor_list[key_col]])
    old_actors = set([k for k in actor_data.data.keys()])
    for old_actor in list(old_actors):
        if old_actor not in new_actors:
            for col in cols:
                if col != key_col and col not in actor_data.data[old_actor]["meta"]:
                    actor_data.data[old_actor]["meta"][col]=""
            for old_col in list(actor_data.data[old_actor]["meta"].keys()):
                if old_col not in set(cols):
                    del actor_data.data[old_actor]["meta"][old_col]
    for i,row in actor_list.iterrows():
        actor = row[key_col]
        if actor in actor_data.data and "data" in actor_data.data[actor]:
            actor_data.data[actor]["meta"]={}
        else:
            actor_data.data[actor]={"meta":{},"data":{}}
        for col in cols:
            if col != key_col:
                actor_data.data[actor]["meta"][col]=row[col]
    if with_full_delete:
        for actor in old_actors:
            if actor not in new_actors and actor not in delete_except:
                del actor_data.data[actor]
    actor_data.simple_update({})

def combine_projects(new_main_path,project_paths):

    new_project = Project(new_main_path,init=True).get_project(format="dict")
    for project_path in project_paths:
        project = Project(project_path,init=True).get_project(format="dict")
        old_new_ref_inters = set(project["referal_data"].data.keys()).intersection(set(new_project["referal_data"].data.keys()))
        if len(old_new_ref_inters) > 0:
            for k in old_new_ref_inters:
                for source, dat in list(new_project["referal_data"].data[k]["data"].items()):
                    if source not in project["referal_data"].data[k]["data"]:
                        project["referal_data"].data[k]["data"][source]=dat
                    else:
                        new_output_keys = set([Spread._get_message_id(data=doc,method=project["referal_data"].data[k]["data"][source]["method"]) for doc in project["referal_data"].data[k]["data"][source]["output"]])
                        for doc in dat["output"]:
                            if Spread._get_message_id(data=doc,method=dat["method"]) not in new_output_keys:
                                project["referal_data"].data[k]["data"][source]["output"].append(doc)
        new_project["referal_data"].data.update(project["referal_data"].data)
        new_project["actor_data"].data.update(project["actor_data"].data)
        new_project["cleaned_urls"].data.update(project["cleaned_urls"].data)
        new_project["alias_data"].data.update(project["alias_data"].data)
    new_project["referal_data"].simple_update({})
    new_project["actor_data"].simple_update({})
    new_project["cleaned_urls"].simple_update({})
    new_project["alias_data"].simple_update({})

def restructure_old_data(main_path):

    referal_data = FlatFile(main_path+"/{0}".format("referal_data.p"))
    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    new_referal_data = FlatFile(main_path+"/{0}".format("referal_data_new.p"))
    new_actor_data = FlatFile(main_path+"/{0}".format("actor_data_new.p"))
    swaped_sources = dict([(value, key) for key, value in gvar.SOURCES.items()])
    swaped_sources["crowdtangle"]="Facebook Page"

    for actor,dat in actor_data.data.items():
        if "data" not in dat:
            new_actor_data.data[actor]={"data":dat}
        else:
            new_actor_data.data[actor]=dat
        if "meta" not in dat:
            new_actor_data.data[actor]["meta"]={}
        if "twitter" in new_actor_data.data[actor]["data"]:
            new_actor_data.data[actor]["data"]["twitter2"] = new_actor_data.data[actor]["data"].pop("twitter")
        for source in list(new_actor_data.data[actor]["data"].keys()):
            if source in swaped_sources:
                new_actor_data.data[actor]["data"][swaped_sources[source]] = new_actor_data.data[actor]["data"].pop(source)

    for url,dat in referal_data.data.items():
        if "data" not in dat:
            new_referal_data.data[url]={"data":dat}
        else:
            new_referal_data.data[url]=dat
        if "meta" not in dat:
            new_referal_data.data[url]["meta"]={}
        if "twitter" in new_referal_data.data[url]["data"]:
            new_referal_data.data[url]["data"]["twitter2"] = new_referal_data.data[url]["data"].pop("twitter")

    new_referal_data.simple_update({})
    new_actor_data.simple_update({})
