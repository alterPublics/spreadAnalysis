from spreadAnalysis.scraper.tw_scraper import TwScraper
from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from datetime import datetime
import random
import sys

class Twitter2:

    def __init__(self,tokens,version=2):

        self.tokens = tokens["tokens"]
        self.version = version
        self.base_url = "https://api.twitter.com/2"

    def _get_headers(self):

        headers =  random.choice(self.tokens)
        return headers

    def _get_exp_params(self):

        self.expansions = ["attachments.poll_ids", "attachments.media_keys", "author_id",
            "entities.mentions.username", "geo.place_id", "in_reply_to_user_id",
            "referenced_tweets.id", "referenced_tweets.id.author_id"]

        self.media_fields = ["duration_ms", "height", "media_key", "preview_image_url",
            "type", "url", "width", "public_metrics"]

        self.place_fields = ["contained_within", "country", "country_code",
            "full_name", "geo", "id", "name", "place_type"]

        self.poll_fields = ["duration_minutes", "end_datetime", "id", "options",
            "voting_status"]

        self.tweet_fields = ["attachments", "author_id", "context_annotations",
            "conversation_id", "created_at", "entities", "geo", "id",
            "in_reply_to_user_id", "lang", "public_metrics", "possibly_sensitive",
            "referenced_tweets", "reply_settings", "source", "text", "withheld"]

        self.user_fields = ["created_at", "description", "entities", "id",
            "location", "name", "pinned_tweet_id", "profile_image_url",
            "protected", "public_metrics", "url", "username", "verified", "withheld"]

        exp_params = {"expansions":','.join(self.expansions),
                        "place.fields":','.join(self.place_fields),
                        "poll.fields":','.join(self.poll_fields),
                        "tweet.fields":','.join(self.tweet_fields),
                        "user.fields":','.join(self.user_fields),
                        }

        return exp_params

    def _user_urls_to_ID(self,urls):

        to_convert = set([str(url) for url in urls if not str(url).isdecimal()])
        to_ignore = set([str(url) for url in urls if not str(url) in to_convert])
        to_convert = set([LinkCleaner().extract_username(url) for url in to_convert])
        to_ignore.update(to_convert)
        #print (to_ignore)
        actor_data = self.actor_info(list(to_ignore))
        #print (actor_data)
        #sys.exit()
        if len(actor_data) < 2:
            user_ids = [dat["id"] for e,dat in actor_data.items()]
        else:
            user_ids = [dat["id"] for e,dat in actor_data.items() if e in to_ignore or e in set([ti.lower() for ti in to_ignore])]
        return user_ids

    def _get_next_token(self,res):
        next_token = None
        if res is not None and res.ok:
            res = res.json()
            if "meta" in res and "next_token" in res["meta"]:
                next_token = res["meta"]["next_token"]
        return next_token

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            res = res.json()
            if "data" in res:
                for e in res["data"]:
                    output_data["output"].append(e)

                if 'includes' in res:
                    if 'tweets' in res["includes"]:
                        tweet_keyed = set([e["id"] for e in output_data["output"]])
                        for e in res["includes"]["tweets"]:
                            if e["id"] not in tweet_keyed:
                                output_data["output"].append(e)

                if 'users' in res["includes"]:
                    users_keyed = {e["id"]:e for e in res["includes"]["users"]}
                    for e in output_data["output"]:
                        if e["author_id"] in users_keyed:
                            e["author"]=users_keyed[e["author_id"]]
                        if "in_reply_to_user_id" in e and isinstance(e["in_reply_to_user_id"],str) and e["in_reply_to_user_id"] in users_keyed:
                            e["in_reply_to_user_id"]=users_keyed[e["in_reply_to_user_id"]]
        return output_data

    def _get_data(self,data,call_url,params,headers,wait_time=0,pag_token_name="next_token",max_results=None):

        res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=30,wait_time=wait_time)
        data = self._update_output(res,data)
        next_token = self._get_next_token(res)
        while next_token:
            params.update({pag_token_name:next_token})
            res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=30,wait_time=wait_time)
            data = self._update_output(res,data)
            next_token = self._get_next_token(res)
            if max_results is not None and len(data["output"]) >= max_results:
                break
        return data

    def actor_info(self,actors):

        actor_data = {}
        call_url = self.base_url+'/users/by'
        for actor_list in hlp.chunks(list(actors),100):
            params = {"usernames":",".join(list(actor_list))}
            params.update({"user.fields":self._get_exp_params()["user.fields"]})
            res = Req.get_response(call_url,params=params,headers=self._get_headers()
                                    ,fail_wait_time=60,wait_time=0.08)

            if res is not None and res.ok:
                actor_data.update({e["username"]:e for e in res.json()["data"]})
        return actor_data

    def url_referals(self,url,start_date=None,end_date=None,wait_time=3.1,max_results=500000,extra_params=None):

        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"twitter2"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        call_url = self.base_url+'/tweets/search/all'
        params = {'query':'"{0}"'.format(url),
                    'start_time':str(start_date)[:10]+'T00:00:00Z',
                    'end_time':str(end_date)[:10]+'T00:00:00Z',
                    'max_results':100}
        params.update(self._get_exp_params())
        if extra_params is not None: params.update(extra_params)
        headers = self._get_headers()

        return self._get_data(data,call_url,params,headers,wait_time=wait_time,max_results=max_results)

    def actor_content(self,actor,start_date=None,end_date=None,max_results=80000):

        actor = LinkCleaner().extract_username(actor,never_none=True)
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"twitter2"}
        call_url = self.base_url+'/tweets/search/all'
        params = {'query':'from:{0}'.format(actor),
                    'start_time':start_date+'T00:00:00Z',
                    'end_time':end_date+'T00:00:00Z',
                    'max_results':100}
        params.update(self._get_exp_params())
        headers = self._get_headers()
        data_to_return = self._get_data(data,call_url,params,headers,wait_time=3.1,max_results=max_results)

        return data_to_return

    def domain_referals(self,domain,start_date=None,end_date=None,full=True,max_results=None):

        new_data = {}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        result_count = 0
        seen_tweet_ids = set([])
        for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=400):
            #extra_params = {'query':'"{0}" (lang:da OR lang:no OR lang:sv OR lang:fi OR lang:de OR lang:nl)'.format(domain)}
            extra_params = {'query':'{0}'.format(domain)}
            dom_data = self.url_referals(domain,start_date=start_date,end_date=end_date,extra_params=extra_params)
            #print (" - ".join([domain,str(start_date),str(end_date),str(len(dom_data["output"]))]))
            for doc in dom_data["output"]:
                new_url = None
                if "entities" in doc and "urls" in doc["entities"]:
                    for url_ent in doc["entities"]["urls"]:
                        url = url_ent["expanded_url"]
                        if LinkCleaner().remove_url_prefix(str(domain)) in str(url):
                            new_url = LinkCleaner().single_clean_url(str(url))
                            break
                if new_url is not None:
                    if new_url not in new_data:
                        new_data.update({new_url:{"input":new_url,
                                    "input_type":"link",
                                    "output":[],
                                    "method":"twitter2"}})
                    if doc["id"] not in seen_tweet_ids:
                        new_data[new_url]["output"].append(doc)
                        seen_tweet_ids.add(doc["id"])
            result_count+=len(dom_data["output"])
            if max_results is not None and result_count > max_results:
                break
        new_data = list(new_data.values())
        return new_data


    def query_content(self,query,start_date=None,end_date=None,max_results=None):

        new_data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"twitter2"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        result_count = 0
        seen_tweet_ids = set([])
        for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=400):
            extra_params = {'query':'{0}'.format(query)}
            dom_data = self.url_referals(query,start_date=start_date,end_date=end_date,extra_params=extra_params)
            for doc in dom_data["output"]:
                if doc["id"] not in seen_tweet_ids:
                    new_data["output"].append(doc)
                    seen_tweet_ids.add(doc["id"])
            result_count+=len(dom_data["output"])
            if max_results is not None and result_count > max_results:
                break
        return new_data

    def actor_content_timeline(self,actor,start_date=None,end_date=None,max_results=None):

        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        if not str(actor).isdecimal():
            actor = self._user_urls_to_ID([actor])[0]
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"twitter2"}
        call_url = self.base_url+'/users/{0}/tweets'.format(actor)
        params = {  'start_time':start_date+'T11:59:59Z',
                    'end_time':end_date+'T00:00:00Z',
                    'max_results':100,
                    'exclude':"replies"}
        params.update(self._get_exp_params())
        headers = self._get_headers()
        data_to_return = self._get_data(data,call_url,params,headers,pag_token_name="pagination_token",wait_time=0.8)

        return data_to_return
