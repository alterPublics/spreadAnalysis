import sys
from spreadAnalysis.utils.link_utils import LinkCleaner
from datetime import datetime
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.io.config_io import Config

PLATFORM_TO_STR_ID = {"facebook":["facebook."],
						"instagram":["instagram."],
						"twitter":["twitter.","t.co"],
						"gab":["gab.com"],
						"tiktok":["tiktok."],
						"telegram":["t.me/"],
						"reddit":["reddit."],
						"youtube":["youtube.","youtu.be"],
						"vkontakte":["vk.com/"]}

try:
	import fasttext
	conf = Config()
	lang_model_gl = fasttext.load_model(conf.LANGDETECT_MODEL)
except:
	pass

class Spread:

	@staticmethod
	def clean_url(url):
		new_url = str(url).strip().rstrip()
		if str(new_url)[-1] == "/": new_url = str(url)[:-1]
		new_url = str(new_url).split("/")[-1]
		if len(new_url) < 2: new_url = str(url).split("/")[-2]
		if "-" in new_url: new_url = new_url.split("-")[-1].strip()

		return new_url

	@staticmethod
	def _get_message_id(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			_val = str(data["platformId"])
		if method=="twitter2":
			_val = str(data["id"])
		if method=="crowdtangle_app":
			_val = str(data["post_url"].split("/")[-1])
		if method=="google":
			_val = str(data["link"])
		if method=="facebook_browser":
			_val = str(data["id"])
		if method=="vkontakte":
			if "post_id" in data:
				_val = str(data["post_id"])+"_"+str(data["id"])
			else:
				_val = str(data["owner_id"])+"_"+str(data["id"])
		if method=="reddit":
			_val = str(data["id"])
		if method=="majestic":
			_val = str(data["SourceURL"])+str(data["TargetURL"])
		if method=="youtube":
			_val = str(data["id"])
		if method=="telegram":
			_val = str(data["id"])+"_"+str(data["peer_id"]["channel_id"])
		if method=="tiktok":
			_val = str(data["id"])
		if method=="gab":
			_val = str(data["id"])
		if method=="fourchan":
			_val = str(data["num"])+"_"+str(data["thread_num"])
		return _val

	@staticmethod
	def _get_message_text(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			text_fields = [str(data[field]) for field in ["title","caption","description","message"] if field in data]
			_val = str(" ".join(text_fields))
		if method=="twitter2":
			_val = str(data["text"])
		if method=="crowdtangle_app":
			_val = str(data["message"])
		if method=="google":
			_val = " ".join([str(data["title"]),str(data["snippet"])])
		if method=="vkontakte":
			_val = str(data["text"])
		if method=="reddit":
			if "selftext" in data:
				_val = str(data["selftext"])
			elif "body" in data:
				_val = str(data["body"])
			else:
				_val = ""
		if method=="majestic":
			_val = str(data["SourceTitle"])
		if method=="youtube":
			text_fields = [str(data["snippet"][field]) for field in ["title","description"] if field in data["snippet"]]
			_val = " ".join(text_fields)
		if method=="telegram":
			if "message" in data:
				_val = str(dict(data)["message"])
			else:
				_val = ""
		if method=="tiktok":
			_val = str(data["desc"])
		if method=="gab":
			_val = str(data["body"])
		if method=="fourchan":
			#print (data)
			if "comment" in data:
				_val = str(data["comment"])
			else:
				_val = ""
		return _val

	@staticmethod
	def _get_message_link(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			if "link" in data:
				_val = str(data["link"])
			else:
				url_list = LinkCleaner().get_url_list_from_text(Spread._get_message_text(data=data,method=method))
				if len(url_list) > 0:
					_val = url_list[0]
		if method=="twitter2":
			if "entities" in data and "urls" in data["entities"]:
				for url_dat in data["entities"]["urls"]:
					if "twitter." not in url_dat["expanded_url"]:
						_val = url_dat["expanded_url"]
						break
					_val = url_dat["expanded_url"]
		if method=="crowdtangle_app":
			_val = str(data["link"])
		if method=="google":
			_val = str(data["link"])
		if method=="vkontakte" or method=="reddit":
			url_list = LinkCleaner().get_url_list_from_text(Spread._get_message_text(data=data,method=method))
			if len(url_list) > 0:
				_val = url_list[0]
		if method=="majestic":
			_val = str(data["SourceURL"])
		if method=="youtube":
			_val = "https://www.youtube.com/watch?v={0}".format(Spread._get_message_id(data=data,method=method))
		if method=="telegram":
			if "media" in data and data["media"] is not None and "webpage" in data["media"] and "url" in data["media"]["webpage"]:
				_val = data["media"]["webpage"]["url"]
			else:
				url_list = LinkCleaner().get_url_list_from_text(Spread._get_message_text(data=data,method=method))
				if len(url_list) > 0:
					_val = url_list[0]
		if method=="tiktok":
			if "playAddr" in data["video"]:
				_val = str(data["video"]["playAddr"])
		if method=="gab":
			url_list = LinkCleaner().get_url_list_from_text(Spread._get_message_text(data=data,method=method))
			if len(url_list) > 0:
				_val = url_list[0]
		if method=="fourchan":
			url_list = LinkCleaner().get_url_list_from_text(Spread._get_message_text(data=data,method=method))
			if len(url_list) > 0:
				_val = url_list[0]
		return _val

	@staticmethod
	def _get_all_external_message_links(method=None,data=None):

		all_text = Spread._get_message_text(method=method,data=data)
		url_list = LinkCleaner().get_url_list_from_text(all_text)
		pl = Spread._get_platform(method=method,data=data)
		all_urls = []
		for url in url_list:
			pl_found = False
			if pl in PLATFORM_TO_STR_ID:
				for pstr in PLATFORM_TO_STR_ID[pl]:
					if pstr in str(url):
						pl_found = True
			if not pl_found:
				all_urls.append(url)

		return all_urls

	@staticmethod
	def _get_message_link_domain(method=None,data=None):
		_val = None
		try:
			_val = LinkCleaner().extract_special_url(Spread._get_message_link(data=data,method=method))
		except:
			LinkCleaner().extract_domain(Spread._get_message_link(data=data,method=method))
		return _val

	@staticmethod
	def _get_date(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			_val = str(data["date"])
		if method=="twitter2":
			_val = str(data["created_at"]).replace("T"," ").split(".")[0]
		if method=="crowdtangle_app":
			_val = str(data["post_date"]).replace("T"," ")
		if method=="google":
			if "pagemap" in data and "metatags" in data["pagemap"]:
				for tag in data["pagemap"]["metatags"]:
					if "article:published_time" in tag:
						_val = tag["article:published_time"].replace("T"," ").split("+")[0]
						break
		if method=="vkontakte":
			_val = str(datetime.fromtimestamp(int(data["date"])))
		if method=="reddit":
			_val = str(datetime.fromtimestamp(int(data["created_utc"])))
		if method=="majestic":
			_val = str(data["FirstIndexedDate"])
		if method=="tiktok":
			_val = str(datetime.fromtimestamp(int(data["createTime"])))
		if method=="youtube":
			_val = str(data["snippet"]["publishedAt"]).replace("T"," ").replace("Z","")
		if method=="telegram":
			_val = str(data["date"])[:19]
		if method=="gab":
			_val = str(data["created_at"]).replace("T"," ").split(".")[0]
		if method=="fourchan":
			if "timestamp" in data:
				_val = str(datetime.fromtimestamp(int(data["timestamp"])))
		return _val

	@staticmethod
	def _get_actor_id(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			if "platformId" in data["account"]:
				_val = str(data["account"]["platformId"])
			else:
				val = Spread()._get_actor_username(method=method,data=data)
		if method=="twitter2":
			_val = str(data["author"]["id"])
		if method=="crowdtangle_app":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="google":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="facebook_browser":
			return str(data["user_id"])
		if method=="vkontakte":
			_val = data["owner_id"]
		if method=="reddit":
			if "author_fullname" in data:
				_val = data["author_fullname"]
			else:
				_val = data["author"]
		if method=="majestic":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="youtube":
			_val = str(data["snippet"]["channelId"])
		if method=="telegram":
			_val = str(data["peer_id"]["channel_id"])
		if method=="tiktok":
			_val = str(data["author"]["id"])
		if method=="gab":
			_val = str(data["account"]["id"])
		if method=="fourchan":
			_val = str(data["thread_num"])

		return _val

	@staticmethod
	def _get_actor_username(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			if "handle" in data["account"]:
				_val = str(data["account"]["handle"])
				if len(_val) < 2:
					_val = LinkCleaner().extract_username(str(data["account"]["url"]))
			else:
				_val = LinkCleaner().extract_username(str(data["account"]["url"]))
		if method=="twitter2":
			_val = str(data["author"]["username"])
		if method=="crowdtangle_app":
			if str(data["type_id"]) == str("11"):
				_val = str(data["name"]).split(" ")[0]
			_val = str(data["post_url"].split("/")[-1])
		if method=="google":
			#_val = LinkCleaner().extract_domain(str(data["formattedUrl"]))
			#if _val is None:
			_val = str(data["displayLink"])
		if method=="facebook_browser":
			return str(data["user_id"])
		if method=="vkontakte":
			if "screen_name" in data["actor"]:
				_val = data["actor"]["screen_name"]
			elif "id" in data["actor"]:
				_val = data["actor"]["id"]
			else:
				_val = Spread._get_message_id(data=data,method=method)
		if method=="reddit":
			if "author_fullname" in data:
				_val = data["author_fullname"]
			else:
				_val = data["author"]
		if method=="majestic":
			_val = LinkCleaner().extract_domain(str(data["SourceURL"]))
		if method=="youtube":
			if "actor" in data:
				_val = str(data["actor"]["snippet"]["title"])
			elif "snippet" in data and "channelTitle" in data["snippet"]:
				_val = str(data["snippet"]["channelTitle"])
			else:
				print (data)
		if method=="telegram":
			_val = str(data["from_username"])
		if method=="tiktok":
			_val = str(data["author"]["uniqueId"])
		if method=="gab":
			_val = str(data["account"]["username"])
		if method=="fourchan":
			_val = str(data["thread_num"])

		return _val

	@staticmethod
	def _get_actor_name(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			_val = str(data["account"]["name"])
			if len(str(_val)) < 1:
				_val = Spread()._get_actor_username(method=method,data=data)
		if method=="twitter2":
			_val = str(data["author"]["name"])
		if method=="crowdtangle_app":
			_val = str(data["name"])
		if method=="google":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="facebook_browser":
			return str(data["name"])
		if method=="vkontakte":
			if "first_name" in data["actor"]:
				_val = data["actor"]["first_name"]+" "+data["actor"]["last_name"]
			elif "name" in data["actor"]:
				 _val = data["actor"]["name"]
			else:
				_val = Spread()._get_actor_username(method=method,data=data)
		if method=="reddit":
			_val = data["author"]
		if method=="majestic":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="youtube":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="telegram":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="tiktok":
			_val = str(data["author"]["nickname"])
		if method=="gab":
			_val = str(data["account"]["display_name"])
		if method=="fourchan":
			if "name" in data:
				_val = str(data["name"])

		return _val

	@staticmethod
	def _get_followers(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			_val = int(data["account"]["subscriberCount"])
		if method=="twitter2":
			_val = int(data["author"]["public_metrics"]["followers_count"])
		if method=="crowdtangle_app":
			_val = int(data["page_size"])
		if method=="google":
			_val = None
		if method=="vkontakte":
			_val = None
		if method=="reddit":
			_val = None
		if method=="majestic":
			_val = int(data["SourceTrustFlow"])
		if method=="youtube":
			if "subscriberCount" in data:
				_val = int(data["actor"]["statistics"]["subscriberCount"])
		if method=="telegram":
			_val = None
		if method=="tiktok":
			_val = None
		if method=="gab":
			if "followers_count" in data:
				_val = int(data["followers_count"])
		if method=="fourchan":
			_val = None

		return _val

	@staticmethod
	def _get_platform(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			_val = str(data["platform"]).lower()
		if method=="twitter2":
			_val = "twitter"
		if method=="crowdtangle_app":
			if str(data["type_id"]) == "1": _val = "facebook"
			elif str(data["type_id"]) == "2": _val = "facebook"
			elif str(data["type_id"]) == "3": _val = "facebook"
			elif str(data["type_id"]) == "5": _val = "twitter"
			elif str(data["type_id"]) == "8": _val = "instagram"
			elif str(data["type_id"]) == "11": _val = "reddit"
		if method=="google":
			_val = "web"
		if method=="facebook_browser":
			_val = "facebook"
		if method=="vkontakte":
			_val = "vkontakte"
		if method=="reddit":
			_val = "reddit"
		if method=="majestic":
			_val = "web"
		if method=="youtube":
			_val = "youtube"
		if method=="telegram":
			_val = "telegram"
		if method=="tiktok":
			_val = "tiktok"
		if method=="gab":
			_val = "gab"
		if method=="fourchan":
			_val = "fourchan"

		return _val

	@staticmethod
	def _get_account_type(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			if data["platform"]=="Instagram":
				_val = "Instagram_profile"
			else:
				_val = str(data["account"]["accountType"])
		if method=="twitter2":
			_val = "twitter_account"
		if method=="crowdtangle_app":
			if str(data["type_id"]) == "1": _val = "facebook_profile"
			elif str(data["type_id"]) == "2": _val = "facebook_group"
			elif str(data["type_id"]) == "3": _val = "facebook_page"
			elif str(data["type_id"]) == "5": _val = "twitter"
			elif str(data["type_id"]) == "8": _val = "instagram"
			elif str(data["type_id"]) == "11": _val = "reddit"
		if method=="google":
			_val = "web"
		if method=="facebook_browser":
			_val = "facebook_profile"
		if method=="vkontakte":
			_val = "vkontakte"
		if method=="reddit":
			_val = "reddit"
		if method=="majestic":
			_val = "web"
		if method=="youtube":
			_val = "youtube"
		if method=="telegram":
			_val = "telegram"
		if method=="tiktok":
			_val = "tiktok"
		if method=="gab":
			_val = "gab"
		if method=="fourchan":
			_val = "fourchan"

		return _val

	@staticmethod
	def _get_platform_type(method=None,data=None):

		_val = "UNKNOWN"
		if method=="crowdtangle":
			if data["platform"]=="Instagram":
				_val = "Instagram"
			elif "group" in str(data["account"]["accountType"]).lower():
				_val = "Facebook Group"
			else:
				_val = "Facebook Page"
		if method=="twitter2":
			_val = "Twitter"
		if method=="google":
			_val = "Google"
		if method=="facebook_browser":
			_val = "Facebook Profile"
		if method=="vkontakte":
			_val = "Vkontakte"
		if method=="reddit":
			_val = "Reddit"
		if method=="majestic":
			_val = "Majestic"
		if method=="youtube":
			_val = "Youtube"
		if method=="telegram":
			_val = "Telegram"
		if method=="tiktok":
			_val = "Tiktok"
		if method=="gab":
			_val = "Gab"
		if method=="fourchan":
			_val = "Fourchan"

		return _val

	@staticmethod
	def _get_link_to_actor(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			if "url" in data["account"]:
				_val = data["account"]["url"]
			else:
				_val = "https://www.facebook.com/{0}".\
					format(Spread()._get_actor_id(method=method,data=data))
		if method=="twitter2":
			_val = "https://twitter.com/{0}".format(str(data["author"]["username"]))
		if method=="crowdtangle_app":
			_val = str(data["post_url"])
		if method=="google":
			_val = "www."+str(data["displayLink"])
		if method=="facebook_browser":
			_val = "https://www.facebook.com/{0}".\
				format(Spread()._get_actor_id(method=method,data=data))
		if method=="vkontakte":
			_val = "www.vk.com/"+Spread()._get_actor_username(method=method,data=data)
		if method=="reddit":
			_val = Spread()._get_post_url(method=method,data=data)
		if method=="majestic":
			_val = Spread()._get_actor_username(method=method,data=data)
		if method=="youtube":
			_val = "https://www.youtube.com/channel/{0}".format(Spread._get_actor_id(data=data,method=method))
		if method=="telegram":
			_val = "https://t.me/{0}".format(Spread._get_actor_username(data=data,method=method))
		if method=="tiktok":
			_val = "https://www.tiktok.com/@{0}?".format(Spread._get_actor_username(data=data,method=method))
		if method=="gab":
			_val = data["account"]["url"]
		if method=="fourchan":
			_val = "https://boards.4channel.org/w/thread/{0}".format(str(data["thread_num"]))

		return _val

	@staticmethod
	def _get_lang(method=None,data=None,model=None):

		_val = None
		if model is None: model = lang_model_gl
		if method=="crowdtangle":
			if "languageCode" in data:
				_val = str(data["languageCode"])
			else:
				if "message" in data:
					_val = hlp.get_lang_and_conf(str(data["message"]),model=model)["lang"]
				else:
					_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="twitter2":
			_val = str(data["lang"])
		if method=="crowdtangle_app":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="google":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="facebook_browser":
			_val = "da"
		if method=="vkontakte":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="reddit":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="majestic":
			_val = str(data["SourceLanguage"])
		if method=="youtube":
			if "defaultAudioLanguage" in data["snippet"]:
				_val = str(data["snippet"]["defaultAudioLanguage"])
			else:
				_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="telegram":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]
		if method=="tiktok":
			_val = hlp.get_lang_and_conf(str(data["desc"])+" "+str(data["author"]["signature"]),model=model)["lang"]
		if method=="gab":
			_val = str(data["language"])
		if method=="fourchan":
			_val = hlp.get_lang_and_conf(Spread._get_message_text(data=data,method=method),model=model)["lang"]

		return _val

	@staticmethod
	def _get_account_category(method=None,data=None):

		_val = None
		if method=="crowdtangle":
			if data["platform"]=="Instagram":
				_val = "INSTAGRAM_PROFILE"
			elif Spread()._get_account_type(method=method,data=data)=="facebook_profile":
				_val = "PRIVATE_PROFILE"
			elif "pageCategory" not in data["account"] and Spread()._get_account_type(method=method,data=data)=="facebook_group":
				_val = "FACEBOOK_GROUP"
			elif "pageCategory" not in data["account"]:
				_val = None
			else:
				_val = str(data["account"]["pageCategory"])
		if method=="twitter2":
			_val = "twitter_account"
		if method=="crowdtangle_app":
			_val = Spread()._get_account_type(method=method,data=data)
		if method=="google":
			_val = "WEBSITE"
		if method=="facebook_browser":
			if Spread()._get_account_type(method=method,data=data)=="facebook_profile":
				_val = "PRIVATE_PROFILE"
		if method=="vkontakte":
			_val = "vkontakte"
		if method=="reddit":
			_val = "reddit"
		if method=="majestic":
			if "SourceTopicalTrustFlow_Topic_0" in data:
				_val = str(data["SourceTopicalTrustFlow_Topic_0"])
		if method=="youtube":
			_val = str(data["actor"]["kind"])
		if method=="telegram":
			_val = "telegram"
		if method=="tiktok":
			_val = "tiktok"
		if method=="gab":
			_val = "gab"
		if method=="fourchan":
			_val = "fourchan"

		return _val

	@staticmethod
	def _get_interactions(method=None,data=None):

		_val = 0
		if method=="crowdtangle":
			_val = int(sum(list(data["statistics"]["actual"].values())))
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="retweeted":
				_val = int(data["public_metrics"]["like_count"]+data["public_metrics"]["reply_count"])
			else:
				_val = sum(list(data["public_metrics"].values()))
		if method=="crowdtangle_app":
			_val = 0
			fields = ["shares","comments","likes","retweets","love_count",
				"haha_count","wow_count","sad_count","angry_count",
				"thankful_count","care_count"]
			for field in fields:
				_val += int(data[field])
		if method=="google":
			_val = 0
		if method=="vkontakte":
			_val = 0
			for it in ["comments","likes","reposts"]:
				if it in data:
					_val+=int(data[it]["count"])
		if method=="reddit":
			_val = int(data["score"])
		if method=="majestic":
			_val = int(data["SourceCitationFlow"])
		if method=="youtube":
			_val = 0
			fields = ["favoriteCount","commentCount","likeCount","viewCount"]
			for field in fields:
				if field in data["statistics"]:
					_val+=int(data["statistics"][field])
		if method=="telegram":
			if "views" and "forwards" in data and data["views"] is not None and data["forwards"] is not None:
				_val = int(data["views"])+int(data["forwards"])
			else:
				_val = 0
		if method=="tiktok":
			_val = int(data["stats"]["diggCount"])+int(data["stats"]["shareCount"])+int(data["stats"]["commentCount"])+int(data["stats"]["playCount"])
		if method=="gab":
			_val = 0
			fields = ["replies_count","reblogs_count","favourites_count"]
			for field in fields:
				if field in data:
					_val+=int(data[field])
		if method=="fourchan":
			if "op" in data:
				_val = int(data["op"])

		return _val

	def _get_angry(method=None,data=None):

		_val = 0
		if method=="crowdtangle":
			if data["platform"]=="Instagram":
				_val = 0
			else:
				_val = int(data["statistics"]["actual"]["angryCount"])
		if method=="crowdtangle_app":
			_val = int(data["angry_count"])

		return _val

	@staticmethod
	def _get_post_url(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			if "postUrl" not in data:
				_val = Spread()._get_link_to_actor(method=method,data=data)
			else:
				_val = str(data["postUrl"])
		if method=="twitter2":
			_val = "https://twitter.com/any/status/{0}".format(Spread()._get_message_id(method=method,data=data))
		if method=="crowdtangle_app":
			_val = str(data["post_url"])
		if method=="google":
			_val = str(data["link"])
		if method=="vkontakte":
			_val = "https://vk.com/{0}?w=wall{1}_{2}".format(Spread()._get_actor_username(method=method,data=data),Spread()._get_actor_id(method=method,data=data),str(data["id"]))
		if method=="reddit":
			if "full_link" in data:
				_val = data["full_link"]
			elif "permalink" in data:
				_val = "https://www.reddit.com"+data["permalink"]
			else:
				_val = ""
		if method=="majestic":
			_val = str(data["SourceURL"])
		if method=="youtube":
			_val = "https://www.youtube.com/watch?v={0}".format(Spread._get_message_id(data=data,method=method))
		if method=="telegram":
			_val = Spread._get_link_to_actor(data=data,method=method)
		if method=="tiktok":
			_val = "https://www.tiktok.com/@any/video/{0}".format(Spread()._get_message_id(method=method,data=data))
		if method=="gab":
			_val = str(data["url"])
		if method=="fourchan":
			_val = "https://boards.4channel.org/w/thread/{0}#p{1}".format(str(data["thread_num"]),str(data["num"]))

		return _val

	@staticmethod
	def _get_links_in_text(method=None,data=None):

		_val = None
		all_text = Spread._get_message_text(method=method,data=data)
		url_list = LinkCleaner().get_url_list_from_text(all_text)
		message_link = Spread._get_message_link(method=method,data=data)
		if message_link is not None:
			url_list.append(Spread._get_message_link(method=method,data=data))
		if len(url_list) > 0:
			_val = ",".join(url_list)
		return _val

	@staticmethod
	def _get_message_some_mentions(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "t.me/" in url: new_url_list.append(url)
				elif "youtube." in url and "channel/" in url: new_url_list.append(url)
				elif "facebook." in url and "posts/" not in url and "?" not in url: new_url_list.append(url)
				elif "tiktok." in url: new_url_list.append(url)
				elif "vk.com" in url: new_url_list.append(url)
				elif "twitter." in url and "status/" not in url: new_url_list.append(url)

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_link_text(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			text_fields = [str(data[field]) for field in ["title","caption","description"] if field in data]
			_val = " ".join(text_fields)

		return _val

	@staticmethod
	def _get_post_text(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			text_fields = [str(data[field]) for field in ["message"] if field in data]
			_val = " ".join(text_fields)

		return _val

	@staticmethod
	def _get_message_some_mentions(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "t.me/" in url: new_url_list.append(url)
				elif "youtube." in url and "channel/" in url: new_url_list.append(url)
				elif "facebook." in url and "posts/" not in url and "?" not in url: new_url_list.append(url)
				elif "tiktok." in url: new_url_list.append(url)
				elif "vk.com" in url: new_url_list.append(url)
				elif "gab.com" in url: new_url_list.append(url)
				elif "twitter." in url and "status/" not in url: new_url_list.append(url)

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_message_tiktok_mention(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "tiktok." in url: new_url_list.append(url)
				break

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_message_telegram_mention(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "/t.me/" in url:
					#url = "https://t.me/"+url.split("t.me/")[-1].split("/")[0]
					new_url_list.append(url)
					break

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_message_gab_mention(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "/gab.com/" in url:
					#url = "https://t.me/"+url.split("t.me/")[-1].split("/")[0]
					new_url_list.append(url)
					break

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_message_yt_mention(method=None,data=None):

		_val = None
		new_url_list = []
		url_list = Spread._get_links_in_text(method=method,data=data)
		if url_list is not None:
			for url in url_list.split(","):
				if "youtube.com/" in url and "channel/" in url:
					#url = "https://t.me/"+url.split("t.me/")[-1].split("/")[0]
					new_url_list.append(url)
					break

			_val = ",".join(new_url_list)
		return _val

	@staticmethod
	def _get_mentions(method=None,data=None):
		_val = None
		if method=="twitter2":
			if "entities" in data and "mentions" in data["entities"]:
				_val = []
				for m in data["entities"]["mentions"]:
					_val.append(m["username"])
				_val = ",".join(_val)

		return _val

#TWITTER SPECIFIC

	@staticmethod
	def _get_in_reply_to(method=None,data=None):
		_val = None
		if method=="twitter2":
			if "in_reply_to_user_id" in data and "username" in data["in_reply_to_user_id"]:
				_val = data["in_reply_to_user_id"]["username"]
		return _val

	@staticmethod
	def _get_retweet_count(method=None,data=None):

		_val = 0
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="retweeted":
				_val = 0
			else:
				_val = data["public_metrics"]["retweet_count"]
		return _val

	@staticmethod
	def _get_reply_count(method=None,data=None):

		_val = 0
		if method=="twitter2":
			_val = data["public_metrics"]["reply_count"]
		return _val

	@staticmethod
	def _get_favorite_count(method=None,data=None):

		_val = 0
		if method=="twitter2":
			_val = data["public_metrics"]["like_count"]
		return _val

	def _get_quote_count(method=None,data=None):

		_val = 0
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="retweeted":
				_val = 0
			else:
				_val = data["public_metrics"]["quote_count"]
		return _val

	@staticmethod
	def _get_actor_location(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "location" in data["author"]:
				_val = str(data["author"]["location"])
		return _val

	def _get_actor_verified(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "verified" in data["author"]:
				_val = str(data["author"]["verified"])
		return _val

	def _get_actor_followers(method=None,data=None):

		_val = 0
		if method=="twitter2":
			if "followers_count" in data["author"]["public_metrics"]:
				_val = int(data["author"]["public_metrics"]["followers_count"])
		return _val

	def _get_actor_following(method=None,data=None):

		_val = 0
		if method=="twitter2":
			if "following_count" in data["author"]["public_metrics"]:
				_val = int(data["author"]["public_metrics"]["following_count"])
		return _val

	def _get_actor_tweet_count(method=None,data=None):

		_val = 0
		if method=="twitter2":
			if "tweet_count" in data["author"]["public_metrics"]:
				_val = int(data["author"]["public_metrics"]["tweet_count"])
		return _val

	def _get_conversation_id(method=None,data=None):

		_val = None
		if method=="twitter2":
			_val = str(data["conversation_id"])
		return _val

	def _get_is_retweet(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="retweeted":
				_val = True
			else:
				_val = False
		return _val

	def _get_is_reply(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="replied_to":
				_val = True
			else:
				_val = False
		return _val

	def _get_retweet_id(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="retweeted":
				_val = data["referenced_tweets"][0]["id"]
		return _val

	def _get_reply_id(method=None,data=None):

		_val = None
		if method=="twitter2":
			if "referenced_tweets" in data and data["referenced_tweets"][0]["type"]=="replied_to":
				_val = data["referenced_tweets"][0]["id"]
		return _val


#FACEBOOK SPECIFIC


	def _get_message(method=None,data=None):
		_val = ""
		if method=="crowdtangle":
			if "message" in data:
				_val = data["message"]
		return _val

	def _get_caption(method=None,data=None):
		_val = ""
		if method=="crowdtangle":
			if "caption" in data:
				_val = data["caption"]
		return _val

	def _get_title(method=None,data=None):
		_val = ""
		if method=="crowdtangle":
			if "title" in data:
				_val = data["title"]
		return _val

	def _get_description(method=None,data=None):
		_val = ""
		if method=="crowdtangle":
			if "description" in data:
				_val = data["description"]
		return _val

	def _get_subscriber_count(method=None,data=None):
		_val = None
		if method=="crowdtangle":
			if "subscriberCount" in data:
				_val = data["subscriberCount"]
		return _val

	def _get_like_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["likeCount"]
		return _val




	def _get_comment_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["commentCount"]
		return _val

	def _get_share_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["shareCount"]
		return _val

	def _get_angry_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["angryCount"]
		return _val

	def _get_sad_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["sadCount"]
		return _val

	def _get_love_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["loveCount"]
		return _val

	def _get_haha_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["hahaCount"]
		return _val

	def _get_wow_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["wowCount"]
		return _val

	def _get_thankful_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["thankfulCount"]
		return _val

	def _get_care_count(method=None,data=None):
		_val = 0
		if method=="crowdtangle":
			_val = data["statistics"]["actual"]["careCount"]
		return _val
