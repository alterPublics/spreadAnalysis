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

import pandas as pd
import random
import time

class ActorCollection(Collection):

    def __init__(self,
                main_path,
                include_sources=set({"crowdtangle","twitter2","telegram",
                                    "tiktok","vkontakte","youtube"}),
                include_iterations = set(["0"]),
                start_date=None,
                end_date=None):

        self.main_path = main_path
        self.include_sources = include_sources
        self.include_iterations = include_iterations
        self.project = None
        self.start_date = start_date
        self.end_date = end_date

        self.conf = Config()
        self.some = {   "crowdtangle":Crowdtangle(self.conf.get_auth()["crowdtangle"]),
                        "twitter2":Twitter2(self.conf.get_auth()["twitter2"]),
                        "telegram":None,
                        "tiktok":Tiktok(),
                        "vkontakte":Vkontakte(self.conf.get_auth()["vkontakte"]),
                        "youtube":Youtube(self.conf.get_auth()["youtube"]),
                        "crowdtangle_insta":Crowdtangle(self.conf.get_auth()["crowdtangle_insta"])  }
        #Telegram(self.conf.get_auth()["telegram"])

    def _generate_actor_iter(self):

        actor_iter = {}
        if self.project is not None:
            for actor,dat in self.project["actor_data"].data.items():
                if str(dat["meta"]["Iteration"]) in self.include_iterations:
                    actor_iter[actor]={}
                    for source in gvar.SOURCES.keys():
                        if gvar.SOURCES[source] in self.include_sources:
                            if dat["meta"][source] is not None:
                                actor_iter[actor][source]=dat["meta"][source]
        return actor_iter

    def load_project(self,init=True,actor_meta_file="Actors.xlsx"):

        self.project = Project(self.main_path,init=init,actor_meta_file=actor_meta_file).get_project(format="dict")
        wrang.update_actor_meta(self.project["actor_meta_data"],
            self.project["actor_data"],key_col="Actor")
        wrang.update_actor_aliases(self.main_path)
        #wrang.connect_actors_to_urls(self.main_path)

    def collect(self):

        collected = 0
        for actor,sources in self._generate_actor_iter().items():
            at_least_one_collected = False
            print (actor)
            for source,account in sources.items():
                method = self.some[gvar.SOURCES[source]]
                if source not in self.project["actor_data"].data[actor]["data"]:
                    try:
                        account_data = method.actor_content(account,start_date=self.start_date,end_date=self.end_date)
                        self.project["actor_data"].data[actor]["data"][source]=account_data
                        collected += 1
                        at_least_one_collected = True
                        print ("\t {1} : {0}".format(len(self.project["actor_data"].data[actor]["data"][source]["output"]),source))
                        self.project["actor_data"].simple_update({})
                        if source == "tiktok":
                            del self.some["tiktok"]
                            self.some["tiktok"]=Tiktok()
                    except:
                        print ("ERRROR - {0}".format(source))
                        time.sleep(0.5)
            if collected != 0 and collected % 4*len(self.include_sources) == 0 and at_least_one_collected:
                self.project["actor_data"].simple_update({})
            if collected != 0 and collected % 10*len(self.include_sources) == 0 and at_least_one_collected:
                pass
                #self.project["actor_data_backup"].data = {}
                #self.project["actor_data_backup"].simple_update(self.project["actor_data"].data)
