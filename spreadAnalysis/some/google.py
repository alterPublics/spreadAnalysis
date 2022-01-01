from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime
import random
import pandas as pd

class Google:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"

    def _get_next_idx_page(self,res,max_idx=50):

        next_idx = None
        if res is not None and res.ok:
            res = res.json()
            if "queries" in res and "nextPage" in res["queries"]:
                next_idx = res["queries"]["nextPage"][0]["startIndex"]
                if int(next_idx) > max_idx:
                    next_idx = None
        return next_idx

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            if "items" in res.json():
                for e in res.json()["items"]:
                    output_data["output"].append(e)
        return output_data

    def _get_data(self,data,call_url,params,headers,max_results=50):

        res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=180,wait_time=0.1)
        data = self._update_output(res,data)
        next_idx = self._get_next_idx_page(res,max_idx=max_results)
        while next_idx:
            params.update({"start":next_idx})
            res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=180,wait_time=0.1)
            data = self._update_output(res,data)
            next_idx = self._get_next_idx_page(res,max_idx=max_results)
        return data

    def url_referals(self,url):

        cleaner = LinkCleaner()
        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"google"}
        headers = {'User-Agent': self.user_agent,
                    'Accept': 'application/json'}
        creds = random.choice(self.tokens)
        url = cleaner.remove_url_prefix(url)
        if "/" in str(url[-1]): url = str(url)[:-1]
        call_url = self.base_url
        params = {"key":creds["key"],
                    "cx":creds["cx"],
                    "q":'"{0}"'.format(url)}
        data = self._get_data(data,call_url,params,headers)

        return data

    def query_content(self,query,start_date=None,end_date=None,max_results=50):

        data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"google"}
        headers = {'User-Agent': self.user_agent,
                    'Accept': 'application/json'}
        creds = random.choice(self.tokens)
        call_url = self.base_url
        start_date, end_date = hlp.get_default_dates(start_date,None)
        params = {"key":creds["key"],
                    "cx":creds["cx"],
                    "q":'{0}'.format(query),
                    "gl":"dk",
                    "dateRestrict":"d[{}]".format(hlp.get_diff_in_days(start_date,end_date))}
        data = self._get_data(data,call_url,params,headers,max_results=max_results)

        return data
