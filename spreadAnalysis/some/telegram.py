from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from telethon import TelegramClient
import json
import random
import sys
import time

class Telegram:

    _WAIT_BASE = 0.65476

    def __init__(self,auth,auto_init=True):

        #self.lib_path = lib_path
        self.tokens = auth["tokens"]
        if auto_init:
            token = random.choice(self.tokens)
            self.client = TelegramClient('main_session5', "2040", "b18441a1ff607e10a989891a5462e627")
            self.client.start()

    def human_wait(self):

        human_wait = 0
        d_rnd = random.random()
        _add = random.random()+random.random()
        long_wait = 1
        if d_rnd < 0.33:
            long_wait = random.random()*3.3
        elif d_rnd < 0.94 and d_rnd > 0.33:
            long_wait = random.random()*23.4
        else:
            long_wait = random.random()*61.4
        
        human_wait = long_wait+_add+self._WAIT_BASE
        return human_wait

    def _client_init(self):

        token = random.choice(self.tokens)
        self.client = TelegramClient('main_session2', token["api_id"], token["api_hash"])
        self.client.start()

    async def _get_messages_from_username(self,output_data,username,start_date,end_date,max_results=None):
        moduluses = [40,100,200]
        async for message in self.client.iter_messages(username):
            post_data = message.to_dict()
            if hlp.date_is_between_dates(post_data["date"],start_date,end_date):
                post_data["from_username"]=username
                output_data.append(post_data)
            if len(output_data) > max_results:
                break
            if random.choice(moduluses):
                time.sleep(self.human_wait())
        #return output_data

    async def _get_searched_messages(self,output_data,search_term,start_date,end_date,max_results=None):
        async for message in self.client.iter_messages(None,search=search_term):
            post_data = message.to_dict()
            if hlp.date_is_between_dates(post_data["date"],start_date,end_date):
                post_data["search_term"]=search_term
                output_data.append(post_data)
            if len(output_data) > max_results:
                break

    def _get_data(self,data,start_date,end_date,from_username=None,max_results=None,search_term=None):

        if from_username is not None:
            output_data = []
            with self.client:
                self.client.loop.run_until_complete(self._get_messages_from_username(output_data,from_username,start_date,end_date,max_results=max_results))
            data["output"]=output_data
        elif search_term is not None:
            output_data = []
            with self.client:
                self.client.loop.run_until_complete(self._get_searched_messages(output_data,search_term,start_date,end_date,max_results=max_results))
            data["output"]=output_data
        return data

    def actor_content(self,actor,start_date=None,end_date=None):

        actor = LinkCleaner().extract_username(actor,never_none=True)
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"telegram"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        try:
            returned_data = self._get_data(data,start_date,end_date,from_username=actor,max_results=95000)
        except Exception as e:
            print (e)
            if "ResolveUsernameRequest" in str(e):
                secs = str(e).split("wait of ")[-1].split(" ")[0]
                print ("WAITING... {0}".format(secs))
                time.sleep(int(secs)+5)
                return None
        rnd_time = random.random()
        if rnd_time < 0.05:
            time.sleep(self.human_wait())
        else:
            time.sleep(self.human_wait())
        return returned_data

    def query_content(self,query,start_date=None,end_date=None):

        data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"telegram"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)

        return self._get_data(data,start_date,end_date,search_term=query,max_results=65000)
