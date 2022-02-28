from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.analysis.network import BipartiteNet
from spreadAnalysis.utils import helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.analysis.batching import find_urls
import pandas as pd
import time
import random
import numpy as np
import sys
import gc
from multiprocessing import Pool, Manager
from difflib import SequenceMatcher
import Levenshtein

def multi_find_urls(org_urls):

    prev_urls = set(find_urls(selection={},urls=org_urls))
    return prev_urls

def check_url_similarities(dom_sorted_urls,urls):

    url_sims = {}
    for url in urls:
        real_test_url = str(url)
        dom_url = None
        sim = 0.0
        dom = LinkCleaner().extract_special_url(url)
        if dom in dom_sorted_urls:
            for dom_url in dom_sorted_urls[dom]:
                try:
                    url = LinkCleaner().single_clean_url(url)
                    url = LinkCleaner().sanitize_url_prefix(url)
                    dom_url = LinkCleaner().single_clean_url(dom_url)
                    dom_url = LinkCleaner().sanitize_url_prefix(dom_url)
                    url = url.split("#")[0]
                    if url[-4] == "/amp": url = url[:-4]
                    dom_url = dom_url.split("#")[0]
                    if dom_url[-4] == "/amp": dom_url = dom_url[:-4]
                    url = LinkCleaner()._recursive_trim(url)
                    dom_url = LinkCleaner()._recursive_trim(dom_url)
                except:
                    pass
                sim = Levenshtein.ratio(url, dom_url)
                #if sim > 0.93 and sim < 1.0:
                url_sims[real_test_url]={"sim_url":dom_url,"sim_score":sim}

    if len(url_sims) == 1 or len(urls) == 1:
        return dom_url,sim
    else:
        return url_sims

def get_dom_sorted_urls(full=False):

    mdb = MongoSpread()
    dom_sorted_urls = {}
    if full:
        org_urls = [d["Url"] for d in mdb.database["url"].find()]
        random.shuffle(org_urls)
        num_cores = 6
        pool = Pool(num_cores)
        chunked_org_urls = hlp.chunks(org_urls,num_cores)
        results = pool.map(multi_find_urls,chunked_org_urls)
        for result in results:
            for url in result:
                dom = LinkCleaner().extract_special_url(url)
                if dom not in dom_sorted_urls:
                    dom_sorted_urls[dom]=[]
                dom_sorted_urls[dom].append(url)
    else:
        for url_dat in mdb.database["url"].find({"Domain":0}):
            url = url_dat["Url"]
            dom = LinkCleaner().extract_special_url(url)
            if dom not in dom_sorted_urls:
                dom_sorted_urls[dom]=[]
            dom_sorted_urls[dom].append(url)

    mdb.close()
    return dom_sorted_urls

def test():

    mdb = MongoSpread()
    test_urls = set([])
    print ("sorting prev urls by dom")
    dom_sorted_urls = get_dom_sorted_urls(full=True)
    print (len(dom_sorted_urls))
    sys.exit()
    print ("getting url sample")
    #cur = mdb.database["url_bi_network"].aggregate([{"$match": {"entity_type": "{0}".format("url")}},{ "$sample" : { "size": 100000 } }],allowDiskUse=True)
    cur = mdb.database["url_bi_network"].find({},{"url":1}).limit(100000)
    for url_dat in cur:
        test_urls.add(url_dat["url"])
    test_urls = list(test_urls)

    print ("checking similarities")
    check_url_similarities(dom_sorted_urls,test_urls)

def search_urls():

    pass

#test()
