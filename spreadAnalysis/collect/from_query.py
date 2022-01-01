from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.wrangling import wranglers as wrang
from spreadAnalysis import _gvars as gvar
from spreadAnalysis.collect.collection import Collection
from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.some.telegram import Telegram
from spreadAnalysis.some.tiktok import Tiktok
from spreadAnalysis.some.vkontakte import Vkontakte
from spreadAnalysis.some.youtube import Youtube
from spreadAnalysis.some.majestic import Majestic
from spreadAnalysis.some.google import Google
from spreadAnalysis.some.reddit import Reddit

import pandas as pd
import random
import numpy as np

class QueryCollection(Collection):

    def __init__(self,
                main_path,
                include_iterations = set(["0"]),
                include_sources=set({}),
                start_date=None,
                end_date=None,
                only_domains=False):

        self.main_path = main_path
        self.include_iterations = include_iterations
        self.include_sources = include_sources
        self.project = None
        self.start_date = start_date
        self.end_date = end_date
        self.only_domains = only_domains

        self.conf = Config()
        self.some = {   "crowdtangle":Crowdtangle(self.conf.get_auth()["crowdtangle"]),
                        "twitter2":Twitter2(self.conf.get_auth()["twitter2"]),
                        "reddit":Reddit(),
                        "vkontakte":Vkontakte(self.conf.get_auth()["vkontakte"]),
                        "google":Google(self.conf.get_auth()["google"]),
                        "majestic":Majestic(self.conf.get_auth()["majestic"]),  }

    def _generate_url_iter(self):

        url_iter = {}
        if self.project is not None:
            for url,dat in self.project["referal_data"].data.items():
                if str(dat["meta"]["Iteration"]) in self.include_iterations:
                    for source in gvar.SOURCES.values():
                        if source in self.include_sources:
                            if self.only_domains:
                                if dat["meta"]["Domain"] != np.nan and len(str(dat["meta"]["Domain"])) > 0 and int(dat["meta"]["Domain"]) > 0:
                                    if url not in url_iter: url_iter[url]=set([])
                                    url_iter[url].add(source)
                            else:
                                if dat["meta"]["Domain"] != np.nan and len(str(dat["meta"]["Domain"])) > 0 and int(dat["meta"]["Domain"]) == 0:
                                    if url not in url_iter: url_iter[url]=set([])
                                    url_iter[url].add(source)
        return url_iter

    def _add_actor_websites(self,url_iter):

        if self.only_domains:
            for actor,dat in self.project["actor_data"].data.items():
                if str(dat["meta"]["Iteration"]) in self.include_iterations:
                    website = dat["meta"]["Website"]
                    if website is not None and len(str(website)) > 3:
                        for source in gvar.SOURCES.values():
                            if source in self.include_sources:
                                if website not in url_iter: url_iter[website]=set([])
                                url_iter[website].add(source)
        return url_iter

    def load_project(self,init=True):

        self.project = Project(self.main_path,init=init).get_project(format="dict")
        delete_except = set([])
        for domain,sources in self.project["domain_data"].data.items():
            for source,urls in sources.items():
                delete_except.update(set(urls))
        wrang.update_actor_meta(self.project["url_meta_data"],self.project["referal_data"],
                                key_col="Url",with_full_delete=False,delete_except=delete_except)

    def collect(self,add_actor_websites=False,with_update_ref=False):

        #if sources is None: sources = set([s for s in self.some.keys()])
        url_iter = self._generate_url_iter()
        url_iter = list(url_iter.items())
        url_iter.reverse()
        collected = 0
        collected_hits = 0
        backup_count=0
        #prev_cleaned_urls = set([v["unpacked"] for k,v in self.project["cleaned_urls"].data.items()])
        for org_url,sources in url_iter:
            print (org_url)
            at_least_one_collected = False
            if True:
                sources = list(set(sources))
                for source in sources:
                    if not self.only_domains and source in self.project["referal_data"].data[org_url]["data"]:
                        continue
                    if org_url in self.project["domain_data"].data and source in self.project["domain_data"].data[org_url] and not with_update_ref:
                        continue
                    method = self.some[source]
                    #if True:
                    try:
                        if self.only_domains:
                            if with_update_ref:
                                pass
                            elif org_url in self.project["domain_data"].data and source in self.project["domain_data"].data[org_url]:
                                continue
                                #pass
                            referal_data = method.domain_referals(org_url,start_date=self.start_date,end_date=self.end_date)
                            if not org_url in self.project["domain_data"].data:
                                self.project["domain_data"].data[org_url]={source:[k["input"] for k in referal_data]}
                            elif not source in self.project["domain_data"].data[org_url]:
                                self.project["domain_data"].data[org_url][source]=[]
                            self.project["domain_data"].data[org_url][source].extend([k["input"] for k in referal_data])
                            print (self.project["domain_data"].data[org_url].keys())
                            for dom_org_url in referal_data:
                                if dom_org_url["input"] not in self.project["referal_data"].data:
                                    self.project["referal_data"].data[dom_org_url["input"]]={"data":{},"meta":{"Iteration":0,"Domain":0}}
                                if with_update_ref and source in self.project["referal_data"].data[dom_org_url["input"]]["data"]:
                                    prev_ids = set([Spread._get_message_id(data=doc,method=self.project["referal_data"].data[dom_org_url["input"]]["data"][source]["method"]) for doc in self.project["referal_data"].data[dom_org_url["input"]]["data"][source]["output"]])
                                    for doc in dom_org_url["output"]:
                                        if not Spread._get_message_id(data=doc,method=dom_org_url["method"]) in prev_ids:
                                            self.project["referal_data"].data[dom_org_url["input"]]["data"][source]["output"].append(doc)
                                elif source not in self.project["referal_data"].data[dom_org_url["input"]]["data"]:
                                    self.project["referal_data"].data[dom_org_url["input"]]["data"][source]=dom_org_url
                            if collected > 4:
                                self.project["referal_data"].simple_update({})
                                self.project["domain_data"].simple_update({})
                                collected=0
                        else:
                            referal_data = method.url_referals(org_url,start_date=self.start_date,end_date=self.end_date)
                            self.project["referal_data"].data[org_url]["data"][source]=referal_data
                            at_least_one_collected = True
                            collected_hits+=len(referal_data["output"])
                    except Exception as e:
                        print (e)
                        sys.exit()
                    collected += 1
                    if not self.only_domains:
                        try:
                            print ("\t {1} : {0}".format(len(self.project["referal_data"].data[org_url]["data"][source]["output"]),source))
                        except:
                            pass
                if (collected > 360 or collected_hits > 80000) and at_least_one_collected:
                    self.project["referal_data"].simple_update({})
                    collected_hits=0
                    collected = 0
                    backup_count+=1
                if backup_count > 4:
                    #self.project["referal_data_backup"].data = {}
                    #self.project["referal_data_backup"].simple_update(self.project["referal_data"].data)
                    backup_count=0
        self.project["referal_data"].simple_update({})
        self.project["domain_data"].simple_update({})
