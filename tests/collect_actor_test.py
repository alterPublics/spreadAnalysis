from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.persistence.simple import FlatFile
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *
from spreadAnalysis.persistence.schemas import Spread

import pandas as pd
import random

def update_actor_aliases(main_path):

    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    alias_data = FlatFile(main_path+"/{0}".format("alias_data.p"))
    for actor,dat in actor_data.data.items():
        dat = dat["meta"]
        for platform in ["Facebook Page","Facebook Group",
        "Twitter","Instagram","Youtube","TikTok","Vkontakte","Telegram"]:
            if dat[platform] is not None:
                username = LinkCleaner().extract_username(dat[platform])
                if username is not None:
                    alias_data.data[username]=actor
        if dat["Website"] is not None:
            domain = LinkCleaner().\
                extract_domain(dat["Website"])
            alias_data.data[domain]=actor
        alias_data.data[str(actor).lower()]=actor
    alias_data.simple_update({})

def connect_actors_to_urls(main_path):

    referal_data = FlatFile(main_path+"/{0}".format("referal_data.p"))
    actor_data = FlatFile(main_path+"/{0}".format("actor_data.p"))
    alias_data = FlatFile(main_path+"/{0}".format("alias_data.p"))
    domain_to_actor = {dat["meta"]["Website"]:actor for actor,dat in actor_data.data.items() if dat["meta"]["Website"] is not None}
    links_to_own_website = {actor:[] for actor,dat in actor_data.data.items()}
    links_shared_by_actor = {actor:[] for actor,dat in actor_data.data.items()}
    for url,url_dat in list(referal_data.data.items()):
        for domain in list(domain_to_actor.keys()):
            if LinkCleaner().extract_domain(str(domain)) in LinkCleaner().extract_domain(str(url)):
                links_to_own_website[domain_to_actor[domain]].append(url)
        if "total_referals" in url_dat: del url_dat["total_referals"]
        for source in list(url_dat.keys()):
            for post in url_dat[source]["output"]:
                username = Spread._get_actor_username(method=url_dat[source]["method"],data=post)
                actor_id = Spread._get_actor_id(method=url_dat[source]["method"],data=post)
                actor_name = Spread._get_actor_name(method=url_dat[source]["method"],data=post).lower()
                post_link = Spread._get_message_link(method=url_dat[source]["method"],data=post)
                if username in alias_data.data:
                    links_shared_by_actor[alias_data.data[username]].append(url)
                    if post_link is not None: links_shared_by_actor[alias_data.data[username]].append(post_link)
                elif actor_id in alias_data.data:
                    links_shared_by_actor[alias_data.data[actor_id]].append(url)
                    if post_link is not None: links_shared_by_actor[alias_data.data[actor_id]].append(post_link)
                elif actor_name in alias_data.data:
                    links_shared_by_actor[alias_data.data[actor_name]].append(url)
                    if post_link is not None: links_shared_by_actor[alias_data.data[actor_name]].append(post_link)
    for actor,links in links_to_own_website.items():
        actor_data.data[actor]["links_to_own_website"]=list(set(links_to_own_website[actor]))
    for actor,links in links_shared_by_actor.items():
        actor_data.data[actor]["links_shared_by_actor"]=list(set(links_shared_by_actor[actor]))
    actor_data.simple_update({})

def update_actor_meta(actor_list,actor_data,key_col="Actor"):

    cols = actor_list.columns
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
    actor_data.simple_update({})

# SWEDEN
main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/sweden/data_collection"
actor_list_file = main_path+"/{0}".format("Actorlist_SWE.xlsx")
actor_file = main_path+"/{0}".format("actor_data.p")
actor_backup_file = main_path+"/{0}".format("actor_data_backup.p")

# DENMARK
"""main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/denmark/data_collection"
actor_list_file = main_path+"/{0}".format("Actorlist_DK.xlsx")
actor_file = main_path+"/{0}".format("actor_data.p")
actor_backup_file = main_path+"/{0}".format("actor_data_backup.p")"""

actor_data = FlatFile(actor_file)
actor_backup_data = FlatFile(actor_backup_file)

conf = Config()
ct = Crowdtangle(conf.get_auth()["crowdtangle"])
tw = Twitter2(conf.get_auth()["twitter2"])
start_date = "2019-01-01"
include_iterations = set(["0","1"])

actor_list = pd.read_excel(actor_list_file)
actor_list = actor_list.where(pd.notnull(actor_list), None)
update_actor_meta(actor_list,actor_data,key_col="Actor")
update_actor_aliases(main_path)
connect_actors_to_urls(main_path)
actor_data = FlatFile(actor_file)
#print (actor_data.data["Fria Tider"]["data"])
#sys.exit()
"""for actor,dat in actor_data.data.items():
    if "twitter" in dat["data"]:
        for e in dat["data"]["twitter"]["output"]:
            if e["id"] == str("1360341849181937667"):
                print (e)
                sys.exit()
    continue
    print (actor)
    print ()
    #for url in dat["links_to_own_website"]:
        #print ("\t"+" "+url)
    print (len(dat["links_to_own_website"]))
    print (len(set([LinkCleaner().strip_backslash(url) for url in dat["links_to_own_website"]])))
    print ()
    print (actor)
    print ()
    #for url in dat["links_shared_by_actor"]:
        #print ("\t"+" "+url)
    print (len(dat["links_shared_by_actor"]))
    print (len(set([LinkCleaner().strip_backslash(url) for url in dat["links_shared_by_actor"]])))

sys.exit()"""

actor_count = 1
iterate_data = {}
all_actors = list(actor_data.data.items())
#random.shuffle(all_actors)
#all_actors.reverse()
for actor,dat in all_actors:
    print (actor)
    dat = dat["meta"]
    if dat["Iteration"] is not None and str(int(dat["Iteration"])) in include_iterations:
        ct_data = None
        tw_data = None
        tw_count = 0
        fb_count = 0

        # Get Twitter Data
        #if True:
        try:
            if not "twitter" in actor_data.data[actor]["data"]:
                if dat["Twitter"] is not None:
                    tw_user_id = tw._user_urls_to_ID([str(dat["Twitter"])])[0]
                    tw_data = tw.actor_content(tw_user_id,start_date=start_date)
                    tw_count = len(tw_data["output"])
                    #print (tw_count)
                    #sys.exit()
                if tw_data is not None:
                    actor_data.data[actor]["data"]["twitter"]=tw_data
        except:
            print ("ERROR TWITTER")

        # Get Crowdtanlge Data
        if not "crowdtangle" in actor_data.data[actor]["data"]:
            if dat["Facebook Page"] is not None:
                fb_username = LinkCleaner().extract_username(str(dat["Facebook Page"]))
                ct_data = ct.actor_content(fb_username,start_date=start_date)
                fb_count += len(ct_data["output"])
            if dat["Facebook Group"] is not None:
                fb_username = LinkCleaner().extract_username(str(dat["Facebook Group"]))
                if ct_data is None:
                    ct_data = ct.actor_content(fb_username,start_date=start_date)
                    fb_count += len(ct_data["output"])
                else:
                    ct_data["output"].extend(ct.actor_content(fb_username,start_date=start_date)["output"])
                    fb_count = len(ct_data["output"])
            if ct_data is not None:
                actor_data.data[actor]["data"]["crowdtangle"]=ct_data
        if tw_count > 0 or fb_count > 0:
            actor_count += 1
            actor_data.simple_update({})

        print ("Collected from actor: {0} - twitter: {1} - facebook: {2}".format(actor,tw_count,fb_count))
        if actor_count % 5 == 0:
            actor_backup_data.data = {}
            actor_backup_data.simple_update(actor_data.data)
