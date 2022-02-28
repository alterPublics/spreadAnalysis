import sys
import copy
import random
import os
import time
import pandas as pd
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.support.ui import WebDriverWait

from spreadAnalysis.scraper.scraper_helpers import ScraperHelpers

class Scraper(ScraperHelpers):

    USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
                    "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
                    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1","Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:80.0) Gecko/20100101 Firefox/80.0"]

    def __init__(self,settings={}):

        self.settings = settings

        self.browser = None
        self.by_url_data = {}
        self.default_soup_parser = "html.parser"
        self.change_user_agent = False
        self.executable_path = "/usr/bin/chromedriver"
        self.cookie_path = None
        self.cookie_user = ""

        if "change_user_agent" in self.settings:
            self.change_user_agent = self.settings["change_user_agent"]

        if "machine" in self.settings:
            if self.settings["machine"] == "local":
                self.executable_path = "/usr/local/bin/chromedriver"
            else:
                self.executable_path = "/usr/bin/chromedriver 6"

        if "exe_path" in self.settings:
            self.executable_path=self.settings["exe_path"]

    def browser_init(self,user_agent="",proxy=None):

        options = webdriver.ChromeOptions()
        options.add_argument(" - incognito")
        prefs = {"profile.default_content_setting_values.notifications" : 2,
                "profile.managed_default_content_settings.images": 0}
        options.add_experimental_option("prefs",prefs)

        if user_agent != "":
            options.add_argument('--user-agent="{0}"'.format(user_agent))
        if self.change_user_agent:
            options.add_argument(f'--user-agent="{random.choice(self.USER_AGENTS)}"')

        capabilities = DesiredCapabilities.CHROME
        if proxy is not None:
            prox = Proxy()
            prox.proxy_type = ProxyType.MANUAL
            prox.autodetect = False
            prox.http_proxy = proxy
            prox.socks_proxy = proxy
            prox.ssl_proxy = proxy
            options.add_argument('--proxy-server=%s' % proxy)
            options.add_argument("ignore-certificate-errors")

        if "machine" in self.settings and self.settings["machine"] == "local":
            d = DesiredCapabilities.CHROME
            d['goog:loggingPrefs'] = { 'browser':'ALL' }
            self.browser = webdriver.Chrome(executable_path=self.executable_path, chrome_options=options,desired_capabilities=capabilities)
        else:
            options.add_argument('--headless')
            self.browser = webdriver.Chrome(executable_path=self.executable_path, chrome_options=options,desired_capabilities=capabilities)
            self.browser.maximize_window()

        if "cookie_path" in self.settings:
            if "cookie_user" in self.settings:
                self.cookie_user = self.settings["cookie_user"]
            self.cookie_path = self.settings["cookie_path"]
            if not os.path.exists(self.cookie_path):
                os.makedirs(self.cookie_path)
            else:
                try:
                    for cook in pickle.load(open(self.cookie_path+"/cooks_{0}.p".format(self.cookie_user), "rb")):
                        try:
                            self.browser.add_cookie(cook)
                        except:
                            self.browser.get("https://www.{0}".format(str(cook["domain"])[1:]))
                except:
                    print ("cookie file is empty")
                self.browser.refresh()
        self.browser.set_page_load_timeout(30)

    def browser_quit(self):

        if self.cookie_path is not None:
            pickle.dump(self.browser.get_cookies(), open(self.cookie_path+"/cooks_{0}.p".format(self.cookie_user), "wb"))
        self.browser.close()
        time.sleep(5)
        self.browser.quit()

    def browser_reset(self,user_agent="",proxy=None):

        self.browser_quit()
        self.browser_init(user_agent=user_agent,proxy=proxy)

    def finish(self):

        self.browser_quit()
