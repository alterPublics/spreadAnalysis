from urllib.parse import urlparse
import requests
import re
import random
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import signal
import timeout_decorator
from timeout_decorator import TimeoutError
from requests.exceptions import ConnectionError

class LinkUtils:

	USER_AGENTS = [
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"]

	COMMON_SHORTERNERS = set(["http://bit.ly/","https://bit.ly/","http://tinyurl.com/","https://tinyurl.com/",
							"http://goo.gl/","https://goo.gl/","https://t.co/","http://t.co/","buff.ly"])

	COMMON_PROBLEMS = set([".mp4",".mp3"])

	def __init__(self):

		pass

	def single_clean_url(self,url):

		new_url = str(url)
		if "&fbclid" in new_url: new_url = new_url.split("&fbclid")[0]
		if "?fbclid" in new_url: new_url = new_url.split("?fbclid")[0]
		if "&ocid=" in new_url: new_url = new_url.split("&ocid=")[0]
		if "?ocid=" in new_url: new_url = new_url.split("?ocid=")[0]
		if "&feature=youtu.be" in new_url: new_url = new_url.split("&feature=youtu.be")[0]
		if "?feature=youtu.be" in new_url: new_url = new_url.split("?feature=youtu.be")[0]
		if "&feature=" in new_url: new_url = new_url.split("&feature=")[0]
		if "?feature=" in new_url: new_url = new_url.split("?feature=")[0]
		if "&r=" in new_url: new_url = new_url.split("&r=")[0]
		if "?r=" in new_url: new_url = new_url.split("?r=")[0]
		#if "&s=" in new_url: new_url = new_url.split("&s=")[0]
		#if "?s=" in new_url: new_url = new_url.split("?s=")[0]
		if "&cid_source" in new_url: new_url = new_url.split("&cid_source")[0]
		if "?cid_source" in new_url: new_url = new_url.split("?cid_source")[0]
		if "&utm_source" in new_url: new_url = new_url.split("&utm_source")[0]
		if "?utm_source" in new_url: new_url = new_url.split("?utm_source")[0]
		if "&recruiter=" in new_url: new_url = new_url.split("&recruiter=")[0]
		if "?recruiter=" in new_url: new_url = new_url.split("?recruiter=")[0]
		if "?type=3" in new_url: new_url = new_url.split("?type=3")[0]
		if "&type=3" in new_url: new_url = new_url.split("&type=3")[0]
		if "?set=" in new_url: new_url = new_url.split("?set=")[0]
		if "&set=" in new_url: new_url = new_url.split("&set=")[0]
		if "?print=1" in new_url: new_url = new_url.split("?print=1=")[0]
		if "&print=1" in new_url: new_url = new_url.split("&print=1=")[0]

		for r in range(2):
			if str(new_url)[-1] == "]": new_url = str(new_url)[:-1]
			if str(new_url)[-1] == ")": new_url = str(new_url)[:-1]
			if str(new_url)[-1] == ".": new_url = str(new_url)[:-1]
			if str(new_url)[-1] == ",": new_url = str(new_url)[:-1]
			if str(new_url)[-1] == "/": new_url = str(new_url)[:-1]

		if "](" in str(new_url): new_url = new_url.split("](")[0]

		return self._recursive_trim(new_url)

	def _recursive_trim(self,url):

		while url[-1].isalnum() == False:
			url = url[:-1]
		return url

	def sanitize_url_prefix(self,url):

		if "WWW." in url: url = url.replace("WWW.","www.")
		if "HTTPS:" in url: url = url.replace("HTTPS:","https:")
		if "HTTP:" in url: url = url.replace("HTTP:","http:")
		if "www." in url and "http" in url:
			url = url.replace("www.","")
		if "www." in url and "http" not in url:
			url = url.replace("www.","https://")
		if "http:" in url and "https:" not in url:
			url = url.replace("http:","https:")
		return url

	def strip_backslash(self,url):

		if str(url)[-1] == "/":
			url = str(url)[:-1]
		return url

	def get_url_list_from_text(self,inp_text,sanitize=False):


		found_urls = []
		duplicate_urls = set([])
		inp_text = str(inp_text)
		for s_string in ["(?P<url>https?://[^\s]+)","(?P<url>http?://[^\s]+)","(?P<url>www?.[^\s]+)"]:
			matched = re.findall(s_string, inp_text)
			for found_url in matched:
				duplicate_check_url = self.sanitize_url_prefix(found_url)
				if duplicate_check_url not in duplicate_urls:
					if sanitize:
						if "www." in found_url and "http" in found_url:
							found_url = found_url.replace("www.","")
						if found_url not in set(found_urls):
							found_urls.append(self.sanitize_url_prefix(found_url))
					else:
						found_urls.append(found_url)
				duplicate_urls.add(duplicate_check_url)
		return list(found_urls)

	def remove_url_prefix(self,url):

		new_url = url
		if new_url is None: return None
		if len(new_url) < 2: return None
		if "https://" in new_url:
			new_url = new_url.replace("https://","")

		if "http://" in new_url:
			new_url = new_url.replace("http://","")

		if "www." in new_url:
			new_url = new_url.replace("www.","")

		return new_url

	def is_url_domain(self,url):

		new_url = self.remove_url_prefix(url)
		if len(new_url) < 2: return False
		new_url = self.strip_backslash(new_url)

		if new_url == self.extract_domain(url) or new_url == self.extract_domain(str(url).replace("www.","https://")):
			return True
		else:
			return False

	def extract_facebook_url(self,url):

		if "story." in url.split("/")[1] or "photo." in url.split("/")[1]:
			url = url.split("/")[0] + "/" + url.split("fbid=")[1].split("&")[0].strip().split(" ")[0]
		elif "groups" == url.split("/")[1]:
			url =  url.split("/")[0] + "/" + url.split("/")[2].split(" ")[0]
		else:
			url = url.split("/")[0] + "/" + url.split("/")[1].split(" ")[0]

		return url


	def extract_youtube_url(self,url):

		if "channel" in url.split("/")[1] and not "watch?" in url.split("/")[1]:
			url = str("youtube.com" + "/" + "channel/" + url.split("/")[2].strip().split(" ")[0])
		elif "v=" in url:
			url = url
			#url = str("youtube.com"+"/"+"watch?v="+url.split("v=")[1].split("&")[0].strip().split(" ")[0])
		else:
			url = str("youtube.com" + "/" + url.split("/")[1].split(" ")[0])
		return url


	def extract_twitter_url(self,url):

		if "twitter.com/i/web" in url:
			url = url
		else:
			url = str("twitter.com" + "/" + url.split("/")[1].split(" ")[0])
		return url


	def extract_instagram_url(self,url):

		if "instagram.com/p/" in url:
			url = "instagram.com" + "/p/" + url.split("/")[2].split(" ")[0]
		else:
			url = "instagram.com" + "/" + url.split("/")[1].split(" ")[0]

		return url


	def extract_reddit_url(self,url):

		if "reddit.com/comments/" in url:
			url = "reddit.com" + "/" + url.split("/comments/")[1].split("/")[0]
		elif "reddit.com/r/" in url:
			url = "reddit.com" + "/r/" + url.split("/")[2].split(" ")[0]
		else:
			pass

		return url


	def extract_domain(self,url):

		try:
			if "http" in url:
				domain = urlparse(url).netloc
				domain = self.remove_url_prefix(domain)
			else:
				domain = url.replace("www.","")
		except:
			domain = url.split("/")[0]

		return domain

	def extract_special_url(self,full_url):

		if full_url is None: return None
		url = self.remove_url_prefix(full_url)
		if "/" in url:
			if "facebook." in url:
				try:
					special_url = self.extract_facebook_url(url)
				except:
					special_url = self.extract_domain(full_url)
			elif "youtube." in url or "youtu.be" in url:
				try:
					special_url = self.extract_youtube_url(url)
				except:
					special_url = self.extract_domain(full_url)
			elif "twitter." in url:
				special_url = self.extract_twitter_url(url)
			elif "instagram." in url:
				special_url = self.extract_instagram_url(url)
			elif "reddit." in url:
				special_url = self.extract_reddit_url(url)
			elif "t.me/" in url:
				special_url = "https://t.me/"+self.extract_username(url)
			elif "tiktok." in url:
				special_url = "https://tiktok.me/"+self.extract_username(url,with_unpack=False)
			elif "gab.com/" in url:
				special_url = "https://gab.com/"+self.extract_username(url)
			elif "vk.com/" in url:
				special_url = "https://vk.com/"+self.extract_username(url)
			else:
				special_url = self.extract_domain(full_url)
		else:
			special_url = self.extract_domain(full_url)
		return special_url


	@timeout_decorator.timeout(seconds=30)
	def get_url_from_scrape(self,url):

		try:
			scrp = self.scraper
			try:
				scrp.browser.get(url)
				time.sleep(0.25)
			except TimeoutError as e:
				print ("Scraper timed out")
				scrp.browser.quit()
				return url, ""
			except Exception as e:
				print (e)
				scrp.browser.quit()
				return url, ""
			unpacked_url = scrp.browser.current_url
			html = scrp.browser.page_source
			print ("Url extracted using scrape : {0}".format(unpacked_url))
			scrp.browser.quit()
			return unpacked_url, html
		except Exception as e:
			print (e)
			return url, ""


	def extract_title_and_raw_text(self,html,unpacked_url):

		title = unpacked_url
		raw_text = ""
		try:
			titles = [t.text for t in BeautifulSoup(html,"html.parser").find_all("h1")]
			if len(list(titles)) > 0: title = titles[0]
		except Exception as e:
			print (e)
		try:
			raw_texts = [tex.text for tex in BeautifulSoup(html,"html.parser").find_all("p")]
			if len(raw_texts) > 0: raw_text = ' '.join(raw_texts)
		except Exception as e:
			print (e)
		return title, raw_text

	def shortener_in_url(self,url):

		for shortener in self.COMMON_SHORTERNERS:
			if str(shortener) in str(url):
				return True
		return False

	def unpack_url(self,url,force_unpack=False):

		if "http" not in url: url = "http://"+url
		url = self._recursive_trim(url)
		url = self.single_clean_url(url)
		headers = {'User-Agent':random.choice(self.USER_AGENTS)}

		# Get initial response from remote server
		try:
			resp = requests.get(url, allow_redirects=True, timeout=5, headers=headers)
			status_code = resp.status_code
		except ConnectionError:
			status_code = 302
		except Exception as e:
			status_code = 404
			print (e)

		# Host might not allow your request. Check for status code 404 and initiate scrape.
		if status_code == 302:
			return url, None, None
		if status_code == 404:
			if force_unpack:
				unpacked_url, html = self.get_url_from_scrape(url)
			else:
				print ("Simple unpack not possible for url : {}".format(url))
				return url, None, None

		# If request is not rejected, make extra check if unpack was succesful.
		# Check if one of common url-shorterners exist in url-string.
		else:
			unpacked_url = resp.url
			try:
				resp.encoding='utf-8'
				html = str(resp.text)
			except Exception as e:
				print (e)
				html = str(resp.text)
			for shortener in self.COMMON_SHORTERNERS:
				if str(shortener) in str(unpacked_url):
					unpacked_url, html = self.get_url_from_scrape(url)

		title, raw_text = self.extract_title_and_raw_text(html,unpacked_url)
		return unpacked_url, title, raw_text

	def get_clean_urls(self,inp_text,with_unpack=True,force_unique=True):

		url_list = self.get_url_list_from_text(inp_text)
		final_url_list = []
		unpacked_urls_parsed = set({})

		for full_url in url_list:
			for prob in self.COMMON_PROBLEMS:
				if str(prob) in str(full_url):
					continue
			if "." not in self.remove_url_prefix(full_url):
				continue
			if with_unpack or self.shortener_in_url(full_url):
				try:
					unpacked_url, title, raw_text = self.unpack_url(full_url)
					special_url = self.extract_special_url(unpacked_url)
				except Exception as e:
					print (e)
			else:
				unpacked_url = self._recursive_trim(full_url)
				unpacked_url = self.single_clean_url(unpacked_url)
				title = full_url
				raw_text = ""
				special_url = self.extract_special_url(full_url)
			if unpacked_url is None: continue
			if force_unique:
				if unpacked_url in unpacked_urls_parsed: continue
			unpacked_urls_parsed.add(unpacked_url)
			final_url_list.append({"domain":special_url,
			"org_url":full_url,"unpacked":unpacked_url,
			"title":title,"raw_text":raw_text})

		return final_url_list

class LinkCleaner(LinkUtils):

	def __init__(self,scraper=None):

		self.scraper = scraper

	def clean_url(self,input_url,with_unpack=True):
		url_list = self.get_clean_urls(input_url,with_unpack=with_unpack)
		if len(url_list) == 0:
			#print ("Input does not contain a link. Returning None.")
			return None
		elif len(url_list) == 1:
			url_data = url_list[0]
			url_data["org_url"]=input_url
			return url_data
		else:
			print ("There is more than one url in input. Returning as a list.")
			return url_list
		if self.scraper is not None:
			self.scraper.browser.quit()

	def get_urls_from_text(self,input_text):
		url_list = self.get_clean_urls(input_text)
		return url_list

	def extract_username(self,url,never_none=False,with_unpack=True):

		username = None
		url = str(url)
		#if len(str(url)) < 6: return None
		if "twitter." in url and "/" in url:
			username = str(url).split("twitter.")[-1].split("/")[1]
			if "/" in username: username = username.split("/")[0]
		elif "facebook." in url and "/" in url:
			if "?" in url: url = url.split("?")[0]
			new_url = str(url).strip().rstrip()
			if "groups/" in url:
				new_url = new_url.split("groups/")[-1].split("/")[0]
				username = new_url
			else:
				if str(new_url)[-1] == "/": new_url = str(url)[:-1]
				username = str(new_url).split("/")[-1]
				if len(username) < 2: username = str(url).split("/")[-2]
				if "-" in username: username = username.split("-")[-1].strip()
		elif "instagram." in url and "/" in url and "/" in str(url).split("instagram.")[-1]:
			username = str(url).split("instagram.")[-1].split("/")[1]
			if "/" in username: username = username.split("/")[0]
		elif "youtube." in url:
			if "channel/" in url:
				username = str(url).split("channel/")[-1]
			elif "user/" in url:
				 username = str(url).split("user/")[-1]
			elif "/c/" in url:
				 username = str(url).split("/c/")[-1]
			else:
				username = str(url).split("/")[-1]
			if "/" in username: username = username.split("/")[0]
		elif "tiktok." in url:
			if "@" not in url:
				if with_unpack:
					url = self.unpack_url(url,force_unpack=False)[0]
			username = url.split("tiktok.")[-1].split("/")[1].split("?")[0].\
				replace("@","")
			if "/" in username: username = username.split("/")[0]
		elif "t.me" in url:
			if "/s/" in url: url = url.replace("/s/","/")
			username = str(url).split("t.me")[-1].split("/")[1]
			if "/" in username: username = username.split("/")[0]
		elif "vk.com" in url:
			username = str(url).split("vk.com")[-1].split("/")[1]
			if "/" in username: username = username.split("/")[0]
		elif "gab.com" in url:
			username = str(url).split("gab.com")[-1].split("/")[1]
			if "/" in username: username = username.split("/")[0]

		if username is not None:
			username = username.split("?")[0]
			if str(username[-1]) == "/": username = username[:-1]

		if username is None and never_none:
			username = url

		return username
