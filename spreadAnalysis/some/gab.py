from garc import Garc
from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.schemas import Spread

class Gab:

	def __init__(self,tokens=None):

		self.tokens = tokens["tokens"]
		self.client = Garc(user_account=self.tokens[0]["user"],
							user_password=self.tokens[0]["pwd"])

	def _update_output(self,res,output_data,start_date=None,end_date=None):

		for e in res:
			if hlp.date_is_between_dates(Spread._get_date(data=e,method=output_data["method"]),start_date,end_date):
				output_data["output"].append(e)

		return output_data

	def _get_data(self,data,call_method,max_results=None,start_date=None,end_date=None):

		res = call_method(data["input"],gabs_after=str(start_date)[:10])
		data = self._update_output(res,data,start_date=start_date,end_date=end_date)

		return data

	def actor_content(self,actor,start_date=None,end_date=None,max_results=None):

		actor = LinkCleaner().extract_username(actor,never_none=True)
		data = {"input":actor,
				"input_type":"actor",
				"output":[],
				"method":"gab"}
		start_date, end_date = hlp.get_default_dates(start_date,end_date)
		call_method = self.client.userposts
		data = self._get_data(data,call_method,start_date=start_date,end_date=end_date)

		return data
