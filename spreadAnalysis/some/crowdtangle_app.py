from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from datetime import datetime
import random
import pandas as pd

class CrowdtangleApp:

    def __init__(self,tokens,only_reddit=True):

        self.tokens = tokens["tokens"]
        self.base_url = "https://api.crowdtangle.com/ce/"
        self.app_version = "3.0.3"
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
        self.only_reddit = only_reddit

    def _update_output(self,res,output_data,start_date,end_date):

        if res is not None and res.ok:
            for e in res.json()['result']['posts']['posts']:
                if self.only_reddit and "11" not in str(e["type_id"]):
                    continue
                else:
                    if hlp.date_is_between_dates(Spread._get_date(data=e,method=output_data["method"]),start_date,end_date):
                        output_data["output"].append(e)
        return output_data

    def url_referals(self,url,start_date=None,end_date=None):

        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"crowdtangle_app"}
        headers = {'User-Agent': self.user_agent,
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Sec-Fetch-Mode': 'cors'}
        params = {"token":random.choice(self.tokens)["token"],
                    "version":self.app_version,
                    "link":url}
        call_url = self.base_url+"links"
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        res = Req.get_response(call_url,wait_time=0.2,fail_wait_time=80,params=params,headers=headers)
        data = self._update_output(res,data,start_date,end_date)

        return data
