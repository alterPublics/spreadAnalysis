from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime
import random
import pandas as pd

class Crowdtangle:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.base_url = "https://api.crowdtangle.com"

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            if "result" in res.json():
                for e in res.json()["result"]["posts"]:
                    output_data["output"].append(e)
            else:
                print ("NO RESULT")
                print (res.json())
        return output_data

    def _get_next_page(self,res,add_param=None):
        next_page = None
        if res is not None and res.ok:
            res = res.json()
            if "pagination" in res["result"] and "nextPage" in res["result"]["pagination"]:
                next_page = res["result"]["pagination"]["nextPage"]
        if add_param is not None and next_page is not None:
            next_page = next_page + "&{0}".format(add_param)
        return next_page

    def _get_data(self,data,call_url,params,wait_time=0,add_param_to_paginate=None):

        res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=wait_time,retry_except=[21])
        data = self._update_output(res,data)
        next_page = self._get_next_page(res,add_param=add_param_to_paginate)
        while next_page:
            res = Req.get_response(next_page,fail_wait_time=40,wait_time=wait_time,retry_except=[21])
            data = self._update_output(res,data)
            next_page = self._get_next_page(res,add_param=add_param_to_paginate)
        return data

    def url_referals(self,url,start_date=None,end_date=None):

        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"crowdtangle"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        call_url = self.base_url+"/links"
        params = {"link":url,
        			"startDate":start_date,
        			"endDate":end_date,
        			"token":random.choice(self.tokens)["token"],
        			"count":1000,
        			"platforms":"facebook,instagram"}

        return self._get_data(data,call_url,params,wait_time=1)

    def domain_referals(self,domain,start_date=None,end_date=None,full=True,max_results=None,interval=400,only_in_domain_urls=True):

        new_data = {}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        result_count = 0
        for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
            dom_data = self.url_referals(domain,start_date=start_date,end_date=end_date)
            #print (" - ".join([domain,str(start_date),str(end_date),str(len(dom_data["output"]))]))
            for doc in dom_data["output"]:
                new_url = None
                if "link" in doc and LinkCleaner().remove_url_prefix(str(domain)) in str(doc["link"]):
                    new_url = LinkCleaner().single_clean_url(doc["link"])
                elif "expandedLinks" in doc:
                    for expandedlink in doc["expandedLinks"]:
                        if LinkCleaner().remove_url_prefix(str(domain)) in str(expandedlink["original"]):
                            new_url = LinkCleaner().single_clean_url(expandedlink["original"])
                        elif LinkCleaner().remove_url_prefix(str(domain)) in str(expandedlink["expanded"]):
                            new_url = LinkCleaner().single_clean_url(expandedlink["expanded"])
                if new_url is None and not only_in_domain_urls:
                    new_url = LinkCleaner().single_clean_url(doc["link"])
                if new_url is not None:
                    if new_url not in new_data:
                        new_data.update({new_url:{"input":new_url,
                                    "input_type":"link",
                                    "output":[],
                                    "method":"crowdtangle"}})
                    new_data[new_url]["output"].append(doc)
            result_count+=len(dom_data["output"])
            if max_results is not None and result_count > max_results:
                break
        new_data = list(new_data.values())
        return new_data

    def actor_content(self,actor,start_date=None,end_date=None):

        actor = LinkCleaner().extract_username(actor,never_none=True)
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"crowdtangle"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        call_url = self.base_url+"/posts"
        params = {  "startDate":start_date,
                    "endDate":end_date,
                    "token":random.choice(self.tokens)["token"],
                    "count":100,
                    "accounts":actor,
                    "sortBy":"date"}
        for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=364):
            params["startDate"]=start_date
            params["endDate"]=end_date
            data = self._get_data(data,call_url,params,wait_time=1.0,add_param_to_paginate="accounts={}".format(actor))

        return data

    def query_content(self,query,start_date=None,end_date=None,interval=400):

        data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"crowdtangle"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        call_url = self.base_url+"/posts/search"
        for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
            params = {  "startDate":start_date,
                        "endDate":end_date,
                        "token":random.choice(self.tokens)["token"],
                        "count":100,
                        "searchTerm":query,
                        "sortBy":"date"}
            data = self._get_data(data,call_url,params,wait_time=1.0)

        return data
