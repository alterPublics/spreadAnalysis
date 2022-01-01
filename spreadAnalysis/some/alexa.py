from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime
import random
import pandas as pd

# *** UNFINISHED ***

class Alexa:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.base_url = "https://awis.api.alexa.com/api"

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            if "items" in res.json():
                for e in res.json()["items"]:
                    output_data["output"].append(e)
        return output_data

    def _get_data(self,data,call_url,params,headers):

        # *** START FROM HERE ***
        res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=180,wait_time=0.1)
        print (res.text)
        sys.exit()
        data = self._update_output(res,data)
        next_idx = self._get_next_idx_page(res)
        while next_idx:
            params.update({"start":next_idx})
            res = Req.get_response(call_url,params=params,headers=headers,fail_wait_time=180,wait_time=0.1)
            data = self._update_output(res,data)
            next_idx = self._get_next_idx_page(res)
        return data

    def url_referals(self,url):

        cleaner = LinkCleaner()
        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"alexa"}
        headers = {'x-api-key': random.choice(self.tokens)["x-api-key"],
                    'Accept':'application/xml'}
        url = cleaner.remove_url_prefix(url)
        if "/" in str(url[-1]): url = str(url)[:-1]
        call_url = self.base_url+"/"
        params = {"Action":"SitesLinkingIn",
                    "Count":5,
                    "Url":'"{0}"'.format(url),
                    "ResponseGroup":"SitesLinkingIn"}
        data = self._get_data(data,call_url,params,headers)

        return data
