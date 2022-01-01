from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime
import random
import pandas as pd
import urllib.parse

class Majestic:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.base_url = "https://api.majestic.com/api"
        #self.base_url = "https://developer.majestic.com/api"

    def _get_next_idx_page(self,res,add_count=0):

        next_idx = None
        if res is not None and res.ok:
            res = res.json()
            if "DataTables" in res and "BackLinks" in res["DataTables"]:
                next_idx = int(res["DataTables"]["BackLinks"]["Headers"]["Count"])+add_count
                if int(next_idx) >= int(res["DataTables"]["BackLinks"]["Headers"]["AvailableLines"]):
                    next_idx = None
        return next_idx

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            if "DataTables" in res.json() and "BackLinks" in res.json()["DataTables"]:
                for e in res.json()["DataTables"]["BackLinks"]["Data"]:
                    output_data["output"].append(e)
        return output_data

    def _get_data(self,data,call_url,params,max_results=199999):

        cum_count = 0
        res = Req.get_response(call_url,params=params,fail_wait_time=30,wait_time=0.01)
        data = self._update_output(res,data)
        #print ("majestic : "+str(data["input"])+str(" : ")+str(len(data["output"])))
        next_idx = self._get_next_idx_page(res)
        while next_idx:
            cum_count+=int(params["Count"])
            params.update({"From":next_idx})
            res = Req.get_response(call_url,params=params,fail_wait_time=30,wait_time=0.01)
            data = self._update_output(res,data)
            #print ("majestic : "+str(data["input"])+str(" : ")+str(len(data["output"])))
            next_idx = self._get_next_idx_page(res,add_count=cum_count)
            if len(data["output"]) > max_results:
                break
        return data

    def url_referals(self,url,start_date=None,end_date=None,domain_only=False,with_sort=False,with_date_range=False):

        max_results=199999
        cleaner = LinkCleaner()
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"majestic"}
        creds = random.choice(self.tokens)
        if domain_only:
            url = cleaner.extract_domain(url)
            max_results = 299999
        call_url = self.base_url+"/json"
        params = {"app_api_key":creds["app_api_key"],
                    "cmd":"GetBackLinkData",
                    "item":'{0}'.format(url),
                    "datasource":"fresh",
                    "Count":50000,
                    "ShowDomainInfo":1}
        if with_sort:
            params["ConsumeResourcesForAdditionalProcessing"]=1
            params["SortBy1"]="FirstSeen"
            params["SortDir1"]="desc"
            params["FilteringDepth"]=50000
        elif with_date_range:
            params["ConsumeResourcesForAdditionalProcessing"]=1
            params["FilteringDepth"]=50000
            params["Filters"]=urllib.parse.quote('FirstIndexed("gt","{0}") and FirstIndexed("lt","{1}")'.format(start_date,end_date))
            call_url = 'https://api.majestic.com/api/json?app_api_key=E531615F5A8DFF65E5FAB32495D4B42B&cmd=GetBackLinkData&item={2}&datasource=historic&Count=50000&ShowDomainInfo=1&ConsumeResourcesForAdditionalProcessing=1&FilteringDepth=50000&Filters=FirstIndexed("gt","{0}") and FirstIndexed("lt","{1}")'.format(str(start_date)[:10],str(end_date)[:10],url)
            params = {}

        data = self._get_data(data,call_url,params,max_results=max_results)

        return data

    def domain_referals(self,url,start_date=None,end_date=None,with_sort=False,with_date_range=True):

        new_data = {}
        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        result_count = 0
        all_referal_returns = []
        if with_date_range:
            for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=400):
                dom_data = self.url_referals(url,start_date=start_date,end_date=end_date,domain_only=True,with_sort=False,with_date_range=True)
                all_referal_returns.append(dom_data)
                #print (" - ".join([url,str(start_date),str(end_date),str(len(dom_data["output"]))]))
        else:
            all_referal_returns.append(self.url_referals(url,start_date=start_date,end_date=end_date,domain_only=True,with_sort=with_sort))
        for dom_data in all_referal_returns:
            for doc in dom_data["output"]:
                if LinkCleaner().remove_url_prefix(str(LinkCleaner().extract_domain(url))) in str(doc["TargetURL"]):
                    new_url = str(doc["TargetURL"])
                    if new_url is not None:
                        if new_url not in new_data and new_url.replace("https://","http://") not in new_data and new_url.replace("http://","https://") not in new_data:
                            new_data.update({new_url:{"input":new_url,
                                        "input_type":"link",
                                        "output":[],
                                        "method":"majestic"}})
                        if new_url in new_data and doc["SourceURL"] not in set([d["SourceURL"] for d in new_data[new_url]["output"]]):
                            new_data[new_url]["output"].append(doc)
        new_data = list(new_data.values())
        return new_data
