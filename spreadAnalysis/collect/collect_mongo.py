from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.some.telegram import Telegram
from spreadAnalysis.some.tiktok import Tiktok
from spreadAnalysis.some.vkontakte import Vkontakte
from spreadAnalysis.some.youtube import Youtube
from spreadAnalysis.some.majestic import Majestic
from spreadAnalysis.some.google import Google
from spreadAnalysis.some.reddit import Reddit
from spreadAnalysis.some.gab import Gab
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.persistence.schemas import Spread
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime

class CollectMongo:

    collecters = {"crowdtangle":Crowdtangle,
                    "twitter2":Twitter2,
                    "telegram":Telegram,
                    "tiktok":Tiktok,
                    "vkontakte":Vkontakte,
                    "youtube":Youtube,
                    "majestic":Majestic,
                    "google":Google,
                    "reddit":Reddit,
                    "gab":Gab}

    def __init__(self,main_path,low_memory=True):

        self.main_path = main_path
        self.low_memory = low_memory
        self.mdb = MongoSpread()
        self.conf = Config()
        self.MAX_DATE_RANGE_INTERVAL = 60
        self.MIN_DATE_RANGE_INTERVAL = 3
        self.DATE_RANGE_CONV = 3

    def unpack_output_docs(self,data):

        docs = []
        for outer_doc in data:
            for doc in outer_doc["output"]:
                docs.append(doc)
        return docs

    def get_interval_from_returned_data(self,current_interval,data):

        if len(data) < self.DATE_RANGE_CONV * (self.MAX_DATE_RANGE_INTERVAL/2):
            current_interval = int(self.DATE_RANGE_CONV * current_interval)
        elif len(data) > self.DATE_RANGE_CONV * (self.MAX_DATE_RANGE_INTERVAL*1.5):
            current_interval = int(current_interval / (self.DATE_RANGE_CONV/2))
        if current_interval > self.MAX_DATE_RANGE_INTERVAL:
            current_interval = self.MAX_DATE_RANGE_INTERVAL
        return current_interval

    def save_data(self,db,docs,method,prev_keys,id_finder,update_key_col=None,skip_update=False):

        inserts, updates = [], []

        if prev_keys is None or len(prev_keys) < 1:
            if docs is not None:
                if isinstance(id_finder,str):
                    for doc in docs:
                        doc["method"]=method
                        inserts.append(doc)
                    self.mdb.write_many(db,inserts,update_key_col)
                elif isinstance(id_finder,tuple):
                    for doc in docs:
                        inserts.append(doc)
                    self.mdb.write_many(db,inserts,update_key_col)
                else:
                    for doc in docs:
                        _key1 = id_finder(data=doc,method=method)
                        doc["method"]=method
                        doc[update_key_col]=_key1
                        inserts.append(doc)
                    self.mdb.write_many(db,inserts,update_key_col)
        else:
            if docs is not None:
                if prev_keys is None or isinstance(prev_keys,set):
                    for doc in docs:
                        if isinstance(id_finder,str):
                            _key = doc[id_finder]
                        else:
                            _key = id_finder(data=doc,method=method)
                        doc[update_key_col]=_key
                        doc["method"]=method
                        if prev_keys is not None and _key in prev_keys:
                            updates.append(doc)
                        else:
                            inserts.append(doc)
                            prev_keys.add(_key)
                elif isinstance(prev_keys,dict):
                    for doc in docs:
                        if isinstance(id_finder[0],str):
                            _key1 = doc[id_finder[0]]
                        else:
                            _key1 = id_finder[0](data=doc,method=method)
                        if isinstance(id_finder[1],str):
                            _key2 = doc[id_finder[1]]
                        else:
                            _key2 = id_finder[1](data=doc,method=method)
                        if _key1 in prev_keys and _key2 in prev_keys[_key1]:
                            pass
                        else:
                            inserts.append(doc)
                            if _key1 not in prev_keys: prev_keys[_key1]=set([])
                            prev_keys[_key1].add(_key2)

            self.mdb.insert_many(db,inserts)
            if not skip_update: self.mdb.update_many(db,updates,update_key_col)

        return prev_keys

    def resolve_dates_from_pulls(self,attempts,start_date,end_date):

        new_end_date = None
        new_start_date = None
        attempt_sd = [hlp.to_default_date_format(a["start_date"]) for a in attempts if a["returned_posts"] is not None]
        attempt_ed = [hlp.to_default_date_format(a["end_date"]) for a in attempts if a["returned_posts"] is not None]
        if len(attempt_sd) > 0 and len(attempt_ed) > 0:
            min_start_date = min(attempt_sd)
            max_end_date = max(attempt_ed)
            if hlp.to_default_date_format(end_date) > max_end_date:
                new_start_date = str(max_end_date)[:10]
                new_end_date = str(end_date)
            elif hlp.to_default_date_format(start_date) < min_start_date:
                new_end_date = str(min_start_date)[:10]
                new_start_date = str(start_date)
        else:
            new_end_date = end_date
            new_start_date = start_date

        return new_start_date, new_end_date

    def process_pull(self,input,method,input_type,data,start_date,end_date,prev_pulls,extra_fields={}):

        attempt={"returned_posts":None,"start_date":start_date,
                "end_date":end_date,"inserted_at":datetime.now()}
        if data is not None:
            attempt["returned_posts"]=len(data)
        if input in prev_pulls and method in prev_pulls[input]:
            self.mdb.insert_into_nested(self.mdb.database["pull"],attempt,
                [("input",input),("method",method)])
        else:
            new_pull = {"input":input,"method":method,"input_type":input_type,
                        "attempts":[attempt]}
            if len(extra_fields) > 0:
                new_pull.update(extra_fields)
            self.mdb.insert_one(self.mdb.database["pull"],new_pull)

    def get_pulls(self,endpoint):

        prev_pulls = {}
        for d in self.mdb.database["pull"].find({"input_type":{"$eq":endpoint}}):
            if not d["input"] in prev_pulls: prev_pulls[d["input"]]={}
            if not d["method"] in prev_pulls[d["input"]]: prev_pulls[d["input"]][d["method"]]=d
        return prev_pulls

    def get_aliases(self):

        aliases = {}
        for d in self.mdb.get_data_from_db(self.mdb.database["alias"]):
            if not d["actor"] in aliases: aliases[d["actor"]]=[]
            aliases[d["actor"]].append(d)
        return aliases

    def get_clean_url(self,org_url,is_domain=False):

        clean_url = LinkCleaner().clean_url(org_url,with_unpack=True)["unpacked"]
        clean_url = LinkCleaner().strip_backslash(clean_url)
        if LinkCleaner().is_url_domain(clean_url) and not is_domain:
            clean_url = None
        elif is_domain and not LinkCleaner().is_url_domain(clean_url):
            clean_url = "https://"+LinkCleaner().extract_domain(clean_url)
        return clean_url

    def get_methods(self,endpoint):

        methods = {}
        for platform in self.platform_info:
            if platform["working"]:
                if platform[endpoint+"_endpoint"]:
                    if platform["platform_source"] not in methods or platform["platform_dest"] == "Instagram":
                        if "Instagram" in platform["platform_dest"] and endpoint == "actor":
                            method = self.collecters[platform["platform_source"]]\
                                (self.conf.get_auth()["crowdtangle_insta"])
                            methods["crowdtangle_insta"]=method
                        elif platform["platform_dest"] in set(["Tiktok","Reddit"]):
                            method = self.collecters[platform["platform_source"]]()
                            methods[platform["platform_source"]]=method
                        else:
                            method = self.collecters[platform["platform_source"]]\
                                (self.conf.get_auth()[platform["platform_source"]])
                            methods[platform["platform_source"]]=method
        return methods

    def url_collect(self,org_urls,input_sd,input_ed):

        org_urls = set([d["Url"] for d in list(org_urls) if str(d["Domain"]) == "0"])
        cleaned_urls = {doc["url"]:doc["clean_url"] for doc in \
            self.mdb.database["clean_url"].find({},{ "url": 1, "clean_url": 1})}
        if self.low_memory:
            prev_post_ids = None
            prev_url_ids_post_ids = None
        else:
            prev_post_ids = self.mdb.get_keys_from_db(self.mdb.database["post"],use_col="message_id")
            prev_url_ids_post_ids = self.mdb.get_key_pairs_from_db(self.mdb.database["url_post"],"input","message_id")
        prev_pulls = self.get_pulls("url")

        for org_url in org_urls:
            if org_url in cleaned_urls:
                cleaned_url = cleaned_urls[org_url]
            else:
                cleaned_url = self.get_clean_url(org_url)
                cleaned_urls[org_url]=cleaned_url
                self.mdb.insert_one(self.mdb.database["clean_url"],
                    {"url":org_url,"clean_url":cleaned_url})
            if cleaned_url is None:
                continue

            for method_name, method in self.get_methods("url").items():
                start_date, end_date = input_sd, input_ed
                if org_url in prev_pulls and method_name in prev_pulls[org_url]:
                    start_date, end_date = self.resolve_dates_from_pulls(prev_pulls[org_url][method_name]["attempts"],start_date,end_date)
                    if start_date is None:
                        continue
                try:
                    data = list(method.url_referals(cleaned_url,start_date=start_date,end_date=end_date)["output"])
                except:
                    data = None
                self.process_pull(org_url,method_name,"url",data,start_date,end_date,prev_pulls)

                if data is not None:
                    prev_post_ids = self.save_data(self.mdb.database["post"],
                        data,method_name,prev_post_ids,Spread._get_message_id,update_key_col="message_id")
                    prev_url_ids_post_ids = self.save_data(self.mdb.database["url_post"],
                        [{"input":org_url,"message_id":Spread._get_message_id(method=method_name,data=doc)} for doc in data],
                        method_name,prev_url_ids_post_ids,("input","message_id"),update_key_col=("input","message_id"))
                    print ("({0} : {1})".format(str(start_date),str(end_date)) + " - " + str(cleaned_url) + " - " + str(method_name) + " - " + str(len(data)))
                else:
                    print ("({0} : {1})".format(str(start_date),str(end_date)) + " - " + str(cleaned_url) + " - " + str(method_name) + " - " + str("ERROR in Data Retrieval"))

    def actor_collect(self,org_actors,input_sd,input_ed):

        if self.low_memory:
            prev_post_ids = None
            prev_actor_ids_post_ids = None
        else:
            prev_post_ids = self.mdb.get_keys_from_db(self.mdb.database["post"],use_col="message_id")
            prev_actor_ids_post_ids = self.mdb.get_key_pairs_from_db(self.mdb.database["actor_post"],"input","message_id")
        prev_pulls = self.get_pulls("actor")
        aliases = self.get_aliases()
        methods = self.get_methods("actor")
        platform_to_source = {d["platform_dest"]:d["platform_source"] for d in self.platform_info}
        platform_to_source["Instagram"]="crowdtangle_insta"

        for actor_doc in org_actors:
            actor = actor_doc["Actor"]
            for alias in aliases[actor]:
                platform_alias = alias["alias"]
                method = methods[platform_to_source[alias["platform"]]]
                method_name = platform_to_source[alias["platform"]]
                start_date, end_date = input_sd, input_ed
                if actor in prev_pulls and method_name in prev_pulls[actor]:
                    start_date, end_date = self.resolve_dates_from_pulls(prev_pulls[actor][method_name]["attempts"],start_date,end_date)
                    if start_date is None:
                        continue
                try:
                    data = list(method.actor_content(platform_alias,start_date=start_date,end_date=end_date)["output"])
                except:
                    data = None
                self.process_pull(actor,method_name,"actor",data,start_date,end_date,prev_pulls)

                if data is not None:
                    if method_name == "crowdtangle_insta": method_name = "crowdtangle"
                    prev_post_ids = self.save_data(self.mdb.database["post"],
                        data,method_name,prev_post_ids,Spread._get_message_id,update_key_col="message_id")
                    prev_actor_ids_post_ids = self.save_data(self.mdb.database["actor_post"],
                        [{"input":actor,"message_id":Spread._get_message_id(method=method_name,data=doc)} for doc in data],
                        method_name,prev_actor_ids_post_ids,("input","message_id"),update_key_col=("input","message_id"))
                    print ("({0} : {1})".format(str(start_date),str(end_date)) + " - " + str(platform_alias) + " - " + str(method_name) + " - " + str(len(data)))
                else:
                    print ("({0} : {1})".format(str(start_date),str(end_date)) + " - " + str(platform_alias) + " - " + str(method_name) + " - " + str("ERROR in Data Retrieval"))

    def domain_collect(self,org_urls,input_sd,input_ed,actor_web=True):

        org_urls = set([d["Url"] for d in list(org_urls) if str(d["Domain"]) == "1"])
        if actor_web: org_urls.update(set([d["Website"] for d in \
            self.mdb.get_data_from_db(self.mdb.database["actor"])]))
        cleaned_urls = {doc["url"]:doc["clean_url"] for doc in \
            self.mdb.database["clean_url"].find({},{ "url": 1, "clean_url": 1})}
        if self.low_memory:
            prev_post_ids = None
            prev_url_ids_post_ids = None
            prev_domain_ids_url_ids = None
        else:
            prev_post_ids = self.mdb.get_keys_from_db(self.mdb.database["post"],use_col="message_id")
            prev_url_ids_post_ids = self.mdb.get_key_pairs_from_db(self.mdb.database["url_post"],"input","message_id")
            prev_domain_ids_url_ids = self.mdb.get_key_pairs_from_db(self.mdb.database["domain_url"],"input","url")
        prev_pulls = self.get_pulls("domain")

        for org_url in org_urls:
            if org_url in cleaned_urls:
                cleaned_url = cleaned_urls[org_url]
            else:
                cleaned_url = self.get_clean_url(org_url,is_domain=True)
                cleaned_urls[org_url]=cleaned_url
                self.mdb.insert_one(self.mdb.database["clean_url"],
                    {"url":org_url,"clean_url":cleaned_url})
            if cleaned_url is None:
                continue

            for method_name, method in self.get_methods("domain").items():
                start_date, end_date = input_sd, input_ed
                if org_url in prev_pulls and method_name in prev_pulls[org_url]:
                    start_date, end_date = self.resolve_dates_from_pulls(prev_pulls[org_url][method_name]["attempts"],start_date,end_date)
                    if start_date is None:
                        continue
                l_start_date, l_end_date = start_date, start_date
                current_interval = self.MIN_DATE_RANGE_INTERVAL
                while hlp.to_default_date_format(l_end_date) < hlp.to_default_date_format(end_date):
                    if method_name == "majestic":
                        current_interval = 365
                    l_end_date = hlp.get_next_end_date(l_start_date,end_date,interval=current_interval,max_interval=self.MAX_DATE_RANGE_INTERVAL)
                    try:
                        data = list(method.domain_referals(cleaned_url,start_date=l_start_date,end_date=l_end_date))
                        docs = self.unpack_output_docs(data)
                    except:
                        data = None
                    self.process_pull(org_url,method_name,"domain",docs,l_start_date,l_end_date,prev_pulls)

                    if data is not None:
                        for url_doc in data:
                            prev_post_ids = self.save_data(self.mdb.database["post"],
                                url_doc["output"],method_name,prev_post_ids,Spread._get_message_id,update_key_col="message_id")
                            prev_url_ids_post_ids = self.save_data(self.mdb.database["url_post"],
                                [{"input":url_doc["input"],"message_id":Spread._get_message_id(method=method_name,data=doc)} for doc in url_doc["output"]],
                                method_name,prev_url_ids_post_ids,("input","message_id"),update_key_col=("input","message_id"))
                        prev_domain_ids_url_ids = self.save_data(self.mdb.database["domain_url"],
                            [{"input":org_url,"url":url_doc["input"]} for url_doc in data],
                            method_name,prev_domain_ids_url_ids,("input","url"),update_key_col=("input","url"))
                        current_interval = self.get_interval_from_returned_data(current_interval,docs)
                        print ("({0} : {1})".format(str(l_start_date),str(l_end_date)) + " - " + str(cleaned_url) + " - " + str(method_name) + " - " + str(len(docs)))
                    else:
                        print ("({0} : {1})".format(str(l_start_date),str(l_end_date)) + " - " + str(cleaned_url) + " - " + str(method_name) + " - " + str("ERROR in Data Retrieval"))
                        break
                    l_start_date = l_end_date

    def collect(self,endpoint,platform_list=None,start_date=None,end_date=None,skip_existing=True,iterations=[]):

        if endpoint == "domain": call_endpoint = "url"
        else: call_endpoint = endpoint
        if platform_list is None:
            self.platform_info = self.mdb.get_data_from_db(self.mdb.database["platform"])
        else:
            self.platform_info = [doc for doc in self.mdb.get_data_from_db(self.mdb.database["platform"])\
                if doc["platform_dest"] in platform_list]

        if len(iterations) > 0:
            org_inputs = self.mdb.database[call_endpoint].find({ "Iteration": { "$in": iterations } })
        else:
            org_inputs = self.mdb.database[call_endpoint].find({})

        start_date, end_date = hlp.get_default_dates(start_date,end_date)

        if endpoint == "url":
            self.url_collect(org_inputs,start_date,end_date)
        if endpoint == "actor":
            self.actor_collect(org_inputs,start_date,end_date)
        if endpoint == "domain":
            self.domain_collect(org_inputs,start_date,end_date)