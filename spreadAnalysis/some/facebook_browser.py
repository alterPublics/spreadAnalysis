from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.scraper.fb_scraper import FbScraper
from datetime import datetime
import random
import pandas as pd

TRUE_POST_IDS = {"3723706501022403":"3723705994355787",
                    "3671340356235925":"3671340426235918",
                    "490475225249360":"490476621915887",
                    "10221672887505596":"10221672894665775",
                    "10158150384214272":"10158150384259272",
                    "3304293409697565":"3304293659697540",
                    "10158876607278979":"10158876608263979",
                    "10219750113476699":"10219750113956711",
                    "10158542399348928":"10158542399603928",
                    "10157404229951190":"10157404230241190",
                    "10159000189072590":"10159000200522590",
                    "10158886984339381":"10158886984429381",
                    "10221894652461220":"10221894655741302",
                    "10217014589139244":"10217014589419251",
                    "368105024068875":"368105054068872",
                    "1483457651830717":"1483466651829817"}

real_posts = {"https://www.courthousenews.com/court-tears-up-mask-rule-in-germanys-dusseldorf":"10224691028282468",
                "https://off-guardian.org/2020/10/08/who-accidentally-confirms-covid-is-no-more-dangerous-than-flu":"2757614594491360",
                "https://corona-information.dk/planlagt.shtml":"2750802861845267",
                "https://www.youtube.com/watch?v=b8WFlOFnrIE":"10221525515021170"}

class FacebookBrowser:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.fbscraper = FbScraper(settings={"cookie_path":"/private/tmp","exe_path":"/usr/local/bin/chromedriver","machine":"local"})
        self.fbscraper.browser_init()
        self.fbscraper.fb_login(**random.choice(self.tokens))

    def _get_data(self,data,wait_time=0):

        if data["input_type"]=="link":
            post_id = None
            url = data["input"]
            if "facebook." in url:
                if "fbid=" in url:
                    post_id = url.split("fbid=")[-1].split("&")[0]
                elif "posts/" in url:
                    post_id = url.split("posts/")[-1]
                    post_id = url.split("?")[0]
                    post_id = url.split("&")[0]
                elif "photos/a" in url:
                    post_id = url.split("photos/a")[-1].split("/")[1]
            if post_id is not None:
                if post_id in TRUE_POST_IDS:
                    post_id = TRUE_POST_IDS[post_id]
                share_data = self.fbscraper.get_post_shares(post_id,max_shares=2000)
                for share in share_data:
                    data["output"].append(share.as_dict())
            elif url in real_posts:
                share_data = self.fbscraper.get_post_shares(real_posts[url],max_shares=2000)
                for share in share_data:
                    data["output"].append(share.as_dict())

        return data

    def url_referals(self,url,start_date=None,end_date=None):

        data = {"input":url,
                "input_type":"link",
                "output":[],
                "method":"facebook_browser"}

        return self._get_data(data)

    def safe_finish(self):

        self.fbscraper.browser_quit()
