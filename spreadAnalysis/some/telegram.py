from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from telethon import TelegramClient
import json
import random
import sys

class Telegram:

    def __init__(self,auth,auto_init=True):

        #self.lib_path = lib_path
        self.tokens = auth["tokens"]
        if auto_init:
            token = random.choice(self.tokens)
            self.client = TelegramClient('main_session', token["api_id"], token["api_hash"])
            self.client.start()

    def _client_init(self):

        token = random.choice(self.tokens)
        self.client = TelegramClient('main_session', token["api_id"], token["api_hash"])
        self.client.start()

    async def _get_messages_from_username(self,output_data,username,start_date,end_date,max_results=None):
        async for message in self.client.iter_messages(username):
            post_data = message.to_dict()
            if hlp.date_is_between_dates(post_data["date"],start_date,end_date):
                post_data["from_username"]=username
                output_data.append(post_data)
            if len(output_data) > max_results:
                break
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

        return self._get_data(data,start_date,end_date,from_username=actor,max_results=95000)

    def query_content(self,query,start_date=None,end_date=None):

        data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"telegram"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)

        return self._get_data(data,start_date,end_date,search_term=query,max_results=65000)
