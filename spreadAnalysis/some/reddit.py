from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from datetime import datetime
import random
import pandas as pd

class Reddit:

	def __init__(self):

		self.base_url = "https://api.pushshift.io/reddit"

	def _update_output(self,res,output_data):

		if res is not None and res.ok:
			for e in res.json()["data"]:
				output_data["output"].append(e)
		return output_data

	def _get_data(self,data,call_url,params,wait_time=0,add_param_to_paginate=None):

		res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=wait_time)
		data = self._update_output(res,data)

		return data

	def url_referals(self,url,start_date=None,end_date=None):

		data = {"input":url,
				"input_type":"link",
				"output":[],
				"method":"reddit"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		for call_url in [self.base_url+"/search/comment",self.base_url+"/search/submission"]:
			params = {"q":url,
						"before":int(hlp.to_default_date_format(end_date).replace().timestamp()),
						"after":int(hlp.to_default_date_format(start_date).replace().timestamp()),
						"size":500}
			data = self._get_data(data,call_url,params,wait_time=0.19)

		return data

	def domain_referals(self,domain,start_date=None,end_date=None,full=True,max_results=None,interval=400,only_in_domain_urls=True):

		new_data = {}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		initial_results = 100
		while initial_results >= 100:
			if interval < 1: interval = 1
			for sd,ed in hlp.create_date_ranges(start_date,end_date,interval=interval)[:1]:
				dom_data = self.url_referals(domain,start_date=sd,end_date=ed)
				initial_results = len(dom_data["output"])
				if initial_results >= 100:
					interval = int(interval/2)
				else:
					break
				if interval < 4:
					break
		result_count = 0
		for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
			dom_data = self.url_referals(domain,start_date=start_date,end_date=end_date)
			for doc in dom_data["output"]:
				new_url = None
				if "selftext" in doc:
					url_list = LinkCleaner().get_url_list_from_text(str(doc["selftext"]))
				elif "body" in doc:
					url_list = LinkCleaner().get_url_list_from_text(str(doc["body"]))
				if len(url_list) > 0:
					for turl in url_list:
						if turl is not None and only_in_domain_urls and LinkCleaner().remove_url_prefix(str(domain)) in str(turl):
							new_url = LinkCleaner().single_clean_url(turl)
							break
						elif only_in_domain_urls == False and turl is not None:
							new_url = LinkCleaner().single_clean_url(turl)
							break
				if new_url is not None:
					if new_url not in new_data:
						new_data.update({new_url:{"input":new_url,
									"input_type":"link",
									"output":[],
									"method":"reddit"}})
					new_data[new_url]["output"].append(doc)
			result_count+=len(dom_data["output"])
			if max_results is not None and result_count > max_results:
				break
		new_data = list(new_data.values())
		return new_data

	def query_content(self,query,start_date=None,end_date=None,max_results=None,interval=400,only_in_domain_urls=True):

		new_data = {"input":query,
					"input_type":"query",
					"output":[],
					"method":"reddit"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		initial_results = 100
		while initial_results >= 100:
			for sd,ed in hlp.create_date_ranges(start_date,end_date,interval=interval)[:1]:
				dom_data = self.url_referals(query,start_date=sd,end_date=ed)
				initial_results = len(dom_data["output"])
				if initial_results >= 100:
					interval = int(interval/2)
				else:
					break
				if interval < 4:
					break
		result_count = 0
		for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
			dom_data = self.url_referals(query,start_date=start_date,end_date=end_date)
			for doc in dom_data["output"]:
				new_data["output"].append(doc)
			result_count+=len(dom_data["output"])
			if max_results is not None and result_count > max_results:
				break
		return new_data
