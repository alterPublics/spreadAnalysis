from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.schemas import Spread
from datetime import datetime
import random
import sys
from TikTokApi import TikTokApi

class Tiktok:

    def __init__(self):

        self.client = None

    def _update_output(self,res,output_data,start_date=None,end_date=None):

        for e in res:
            print (e["createTime"])
            if hlp.date_is_between_dates(Spread._get_date(data=e,method=output_data["method"]),start_date,end_date):
                output_data["output"].append(e)

        return output_data

    def _get_data(self,data,call_method,params,max_results=None,start_date=None,end_date=None):

        res = call_method(data["input"],**params)
        data = self._update_output(res,data,start_date=start_date,end_date=end_date)

        return data

    def actor_content(self,actor,start_date=None,end_date=None,max_results=None):

        actor = LinkCleaner().extract_username(actor,never_none=True)
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"tiktok"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        params = {"count":20000}
        self.client = TikTokApi.get_instance()
        call_method = self.client.by_username
        data = self._get_data(data,call_method,params,start_date=start_date,end_date=end_date)
        del self.client

        return data

    def query_content(self,query,start_date=None,end_date=None,max_results=None):

        data = {"input":query,
                "input_type":"query",
                "output":[],
                "method":"tiktok"}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        params = {"count":20000,"custom_verifyFp":"verify_kxiwpk7j_CJC5FnW2_Pyz3_40pm_9cMd_Pm5RbwTSrb69"}
        self.client = TikTokApi.get_instance()
        call_method = self.client.by_hashtag
        data = self._get_data(data,call_method,params,start_date=start_date,end_date=end_date)
        del self.client

        return data
