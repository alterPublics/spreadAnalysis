from bs4 import BeautifulSoup
import sys
import time
from selenium.webdriver.common.keys import Keys

from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils import helpers as hlp
from spreadAnalysis.scraper.fb_structs import PostShare
from datetime import datetime
import random

class FbScraper(Scraper):

    BASE_URL = "https://www.facebook.com"
    BASE_URL_MOBILE = "https://m.facebook.com"

    BASE_URL_USE = BASE_URL_MOBILE

    def __init__(self,auto_login=False,settings={}):
        Scraper.__init__(self,settings=settings)

        if auto_login:
            self.browser_init(user_agent=settings["user_agent"],proxy=settings["proxy"])
            self.fb_login(user=settings["user"],pwd=settings["pwd"])

    def fb_login(self,user=None,pwd=None):

        if user is None:
            user = self.settings["user"]
            pwd = self.settings["pwd"]
        facebook_in_cookies = False
        for cook in self.browser.get_cookies():
            if "facebook." in str(cook["domain"]):
                facebook_in_cookies = True
        if not facebook_in_cookies:
            self.browser.implicitly_wait(2)
            self.browser.get(self.BASE_URL)
            self.browser.add_cookie({'domain': 'facebook.com', 'expiry': 1665058589,
                'httpOnly': True, 'name': 'datr', 'path': '/', 'sameSite': 'None',
                'secure': True, 'value': 'GGB8X3FcTJGs0BCk6NTKmvkX'})
            self.browser.implicitly_wait(2)
            self.browser.get(self.BASE_URL)
            self.browser.find_element_by_xpath('//*[@id="email"]').send_keys((Keys.BACKSPACE*30), str(user))
            self.browser.find_element_by_xpath('//*[@id="pass"]').send_keys((Keys.BACKSPACE*30), str(pwd))
            hlp.random_wait()
            try:
                self.browser.find_element_by_xpath('//*[@id="u_0_b"]').click()
            except:
                self.browser.find_element_by_name('login').click()
        else:
            print ("skipping login")
        """hlp.random_wait()
        #self.browser.save_screenshot("login_page0.png")
        html = self.browser.page_source
        self.browser.get(self.BASE_URL_USE+"/"+"916271928570773")
        hlp.random_wait(between=(3,5))
        try:
            login_box_visible = str(self.browser.find_element_by_css_selector("#mobile_login_bar > div._3-rj > a._54k8._56bs._4n43._6gg6._901w._56bu._52jh").text)
        except:
            login_box_visible = None
        try:
            login_box_visible = str(self.browser.find_element_by_id("mobile_login_bar").text)
        except:
            login_box_visible = None
        if login_box_visible is not None and "Log" in login_box_visible:
            print ("probably not logged in in...")
            self.browser.delete_all_cookies()
            print ("All cookies cleared. Manual restart required.")
            self.finish()
            sys.exit()"""

        hlp.random_wait(between=(1,2))

    def get_post_shares(self,post_id_str,max_shares=None):

        share_data = []
        url = "https://m.facebook.com/browse/shares?id={0}".format(post_id_str)
        post_id = post_id_str
        prev_shares = set([])
        self.go_to_url(url)
        hlp.random_wait(between=(2,4))
        self.click_elem_until_disappear(type="class",attr="primarywrap",child_tag="strong",attr_match_pairs=[{"Se mere":None}])
        full_html = self.browser.page_source
        for share_box_class in ["_1uja _xon"]:
            shares_in_view = BeautifulSoup(full_html,self.default_soup_parser).find_all("div",{"class":share_box_class})
            for share_html in shares_in_view:
                share = PostShare(share_html,self.default_soup_parser,post_id=post_id)
                if not share.get_id() in prev_shares:
                    share_data.append(share)
                    prev_shares.add(share.__id__)
                    #print (share.member_id)
                if max_shares is not None and len(share_data) > max_shares:
                    break
            if max_shares is not None and len(share_data) > max_shares:
                break
        return share_data
