from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from datetime import datetime
import random
import pandas as pd

class Vkontakte:

	def __init__(self,tokens):

		self.tokens = tokens["tokens"]
		self.base_url = "https://api.vk.com/method"
		self.actor_info = {}

	def _update_output(self,res,output_data,start_date=None,end_date=None):

		if res is not None and res.ok:
			for e in res.json()["response"]["items"]:
				if hlp.date_is_between_dates(Spread._get_date(data=e,method=output_data["method"]),start_date,end_date):
					output_data["output"].append(e)
		return output_data

	def _get_next_page(self,res,data,params):
		next_page = None
		if res is not None and res.ok:
			res = res.json()
			if "next_from" in res["response"]:
				params["start_from"]=str(res["response"]["next_from"]).split("/")[0]
				next_page = params
			if len(res["response"]["items"]) >= 100 and len(data["output"]) < int(res["response"]["count"]):
				params["offset"]=str(len(data["output"]))
				next_page = params

		return next_page

	def _get_data(self,data,call_url,params,wait_time=0,with_actor_info=True,start_date=None,end_date=None,max_results=6000):

		res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=wait_time)
		data = self._update_output(res,data,start_date=start_date,end_date=end_date)
		next_page = self._get_next_page(res,data,params)
		prev_nresults = 0
		while next_page:
			res = Req.get_response(call_url,params=next_page,fail_wait_time=40,wait_time=wait_time)
			data = self._update_output(res,data,start_date=start_date,end_date=end_date)
			next_page = self._get_next_page(res,data,params)
			nresults = len(data["output"])
			if len(data["output"]) > max_results or nresults <= prev_nresults:
				break
			prev_nresults = nresults

		if with_actor_info:
			for achunk in hlp.chunks(list(set([row["owner_id"] for row in data["output"] if row["owner_id"] not in self.actor_info])),100):
				user_chunk = [str(id_) for id_ in achunk if str(id_).isdecimal()]
				group_chunk = [str(id_).replace("-","") for id_ in achunk if not str(id_).isdecimal()]
				self.update_actor_info(user_chunk,id_type="user_ids",endpoint="/users.get")
				self.update_actor_info(group_chunk)
			new_output = []
			for row in data["output"]:
				if "actor" not in row: row["actor"]={}
				if str(row["owner_id"]) in self.actor_info:
					row["actor"]=self.actor_info[str(row["owner_id"])]
				#print (row["owner_id"])
			#print ()
			#for k in self.actor_info.keys():
				#print (k)
		return data

	def update_actor_info(self,from_screen_names,verbose=False,id_type="group_ids",endpoint="/groups.getById"):

		ids = []
		call_url = self.base_url+endpoint
		fields = 'contacts,description,members_count,city,country,wiki_page,status,can_see_all_posts,screen_name'
		params = {"v":"5.103",
					id_type:",".join(from_screen_names),
					"access_token":random.choice(self.tokens)["access_token"],
					"fields":fields}
		res = Req.get_response(call_url,params=params,fail_wait_time=40)
		if "error" in dict(res.json()):
			call_url = self.base_url+"/users.get"
			del params["group_ids"]
			params["user_ids"]=",".join(from_screen_names)
			res = Req.get_response(call_url,params=params,fail_wait_time=40)
		if "user_ids" in params:
			ids = {row["screen_name"]:row for row in res.json()["response"]}
			for id, row in list(ids.items()):
				row["id"]=str(row["id"])
				ids[id]=row
				ids[row["id"]]=row
			self.actor_info.update(ids)
		else:
			ids = {row["screen_name"]:row for row in res.json()["response"]}
			for id, row in list(ids.items()):
				if "-" not in str(row["id"]):
					row["id"]="-"+str(row["id"])
				ids[id]=row
				ids[str(row["id"])]=row
			self.actor_info.update(ids)
		if verbose:
			print (res.json())
		return self.actor_info

	def url_referals(self,url,start_date=None,end_date=None):

		data = {"input":url,
				"input_type":"link",
				"output":[],
				"method":"vkontakte"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		call_url = self.base_url+"/newsfeed.search"
		params = {"v":"5.103",
					"start_time":hlp.to_default_date_format(start_date).replace().timestamp(),
					"end_time":hlp.to_default_date_format(end_date).replace().timestamp(),
					"access_token":random.choice(self.tokens)["access_token"],
					"count":200,
					"q":url,
					"extended":1,
					"fields":"screen_name"}

		return self._get_data(data,call_url,params,wait_time=1,start_date=start_date,end_date=end_date)

	def domain_referals(self,domain,start_date=None,end_date=None,full=True,max_results=None,interval=400,only_in_domain_urls=True):

		new_data = {}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		result_count = 0
		for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
			dom_data = self.url_referals(domain,start_date=start_date,end_date=end_date)
			for doc in dom_data["output"]:
				new_url = None
				url_list = LinkCleaner().get_url_list_from_text(str(doc["text"]))
				if len(url_list) > 0:
					for turl in url_list:
						new_url = turl
						if turl is not None and only_in_domain_urls and LinkCleaner().remove_url_prefix(str(domain)) in str(turl):
							new_url = LinkCleaner().single_clean_url(new_url)
							break
						elif not only_in_domain_urls and turl is not None:
							new_url = LinkCleaner().single_clean_url(new_url)
							break
				if new_url is not None:
					if new_url not in new_data:
						new_data.update({new_url:{"input":new_url,
									"input_type":"link",
									"output":[],
									"method":"vkontakte"}})
					new_data[new_url]["output"].append(doc)
			result_count+=len(dom_data["output"])
			if max_results is not None and result_count > max_results:
				break
		new_data = list(new_data.values())
		return new_data

	def query_content(self,query,start_date=None,end_date=None,max_results=None,interval=400):

		new_data = {"input":query,
				"input_type":"query",
				"output":[],
				"method":"vkontakte"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		result_count = 0
		for start_date,end_date in hlp.create_date_ranges(start_date,end_date,interval=interval):
			dom_data = self.url_referals(query,start_date=start_date,end_date=end_date)
			for doc in dom_data["output"]:
				new_data["output"].append(doc)
			result_count+=len(dom_data["output"])
			if max_results is not None and result_count > max_results:
				break
		return new_data

	def actor_content(self,actor,start_date=None,end_date=None):

		actor = LinkCleaner().extract_username(actor,never_none=True)
		data = {"input":actor,
				"input_type":"actor",
				"output":[],
				"method":"vkontakte"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		call_url = self.base_url+"/wall.get"
		if str(actor)[0] == "-" or str(actor).isdecimal():
			pass
		else:
			if actor not in self.actor_info:
				self.update_actor_info([actor])
			actor = self.actor_info[actor]["id"]
		params = {  "v":"5.103",
					"filter":"owner",
					"access_token":random.choice(self.tokens)["access_token"],
					"count":100,
					"owner_id":actor}

		return self._get_data(data,call_url,params,start_date=start_date,end_date=end_date)
