from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis import _gvars as gvar
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.utils.network_utils import NetworkUtils
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.scraper.scraper import Scraper
from difflib import SequenceMatcher
from spreadAnalysis.wrangling import wranglers as wrang
from spreadAnalysis.utils import helpers as hlp
from operator import itemgetter
import numpy as np
import time
import math
import sys
import pickle
from collections import defaultdict
import pandas as pd
import requests
from os import listdir
from os.path import isfile, join
from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.wrangling.viewers import export_actor_to_pandas
import spreadAnalysis.export.export as ex
import scipy.spatial
import networkx as nx

def get_post_from_actor_and_link(actor,link,only_first=True):

    mdb = MongoSpread()
    actor_link = mdb.database["url_bi_network"].find_one({"url":link,"actor_platform":actor})
    if actor_link is not None:
        for mid in actor_link["message_ids"]:
            post = mdb.database["post"].find_one({"message_id":mid})
            break

        return ex.get_platform_uniform_post(post)


def get_actor_net_cat_dist(g,node_type="url",cat="main_fringe_sharp",vals=["grey2","mainstream"]):

    val_counts = {n:{v:0.0 for v in vals} for n in list(g.nodes())}
    expected_dist = np.array([1.0/float(len(vals)) for r in vals])
    for n in list(g.nodes()):
        for o,e in g.edges(n):
            if cat in g.nodes[e]:
                if g.nodes[e][cat] in val_counts[n]:
                    val_counts[n][g.nodes[e][cat]]+=1.0
    for n in list(val_counts.keys()):
        val_counts[n]={k:v/float(np.sum(np.array(list(val_counts[n].values())))) if float(np.sum(np.array(list(val_counts[n].values())))) > 0.0 else 0.0 for k,v in val_counts[n].items()}
        if float(np.sum(np.array(list(val_counts[n].values())))) <= 0.0:
            val_counts[n][cat+"_distance"]=0.0
        else:
            val_counts[n][cat+"_distance"]=1-scipy.spatial.distance.cosine(np.array(list([v for k,v in sorted(val_counts[n].items())])),expected_dist)
    return val_counts

def get_actor_net_exploration(actors=[],degs=0):

    def null_safe_str(v):

        if v is None:
            return ""
        else:
            return v

    mdb = MongoSpread()
    url_n_platforms = {}
    degree0_docs = []
    degree1_docs = []
    degree0_links = []
    org_doms = set([])
    for actor in actors:
        for d0 in mdb.database["url_bi_network"].find({"actor":actor}):
            degree0_docs.append(d0)
            degree0_links.append(d0["url"])

    if degs > 0:
        print(len(degree0_links))
        degree1_inter_docs = {}
        for d0a in degree0_links:
            temp_inter = set([])
            for d1 in mdb.database["url_bi_network"].find({"actor":d0a}):
                if not d1["url"] in degree1_inter_docs: degree1_inter_docs[d1["url"]]=list([])
                if d1["actor"] in degree0_actors:
                    degree1_inter_docs[d1["url"]].append(d1)

        print (len(degree1_inter_docs))
        for url in list(degree1_inter_docs.keys()):
            if len(degree1_inter_docs[url]) >= 2:
                plforms = set([d["platform"] for d in degree1_inter_docs[url]])
                url_n_platforms[url]=plforms
            else:
                del degree1_inter_docs[url]

        print (len(degree1_inter_docs))
        if len(degree1_inter_docs) > 1000000000:
            accept_url = random.sample(list(degree1_inter_docs.keys()),100)
        else:
            accept_url = list(degree1_inter_docs.keys())

        for url in accept_url:
            skip = False
            for dom in org_doms:
                if dom in str(url):
                    skip = True
            if skip: continue
            if len(url_n_platforms[url]) >= 2 and "gab" in url_n_platforms[url] or "vkontakte" in url_n_platforms[url] or "reddit" in url_n_platforms[url] or "youtube" in url_n_platforms[url] or "instagram" in url_n_platforms[url] or "fourchan" in url_n_platforms[url]:
                for d in degree1_inter_docs[url]:
                    degree1_docs.append(d)

    g = nx.Graph()
    for degs in [degree0_docs,degree1_docs]:
        print (len(degs))
        for doc in degs:
            if doc["url"] is not None and doc["actor_platform"] is not None:
                edoc = mdb.database["actor_metric"].find_one({"actor_platform":doc["actor_platform"]})
                g.add_node(doc["url"],type="url",domain=null_safe_str(doc["domain"]),label=doc["url"])
                g.add_node(doc["actor_platform"],type="actor",label=null_safe_str(edoc["actor_name"]),platform=null_safe_str(edoc["platform"]))
                g.add_edge(doc["url"],doc["actor_platform"])
    print (len(g.nodes()))
    g = NetworkUtils().filter_by_degrees(g,degree=2,skip_nodes={},preserve_skip_node_edges=True,extra=None)
    print (len(g.nodes()))
    return g
    #nx.write_gexf(g,"/home/alterpublics/projects/altmed/focus_net_test.gexf")
