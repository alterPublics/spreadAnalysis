from os import environ, path
from dotenv import load_dotenv
import json

class Config:

    def __init__(self):

        self.load_env()
        self.set_env()

    def load_env(self):

        basedir = "/".join(str(path.abspath(path.dirname(__file__)))\
            .split("/")[:-1])
        load_dotenv(path.join(basedir+"/config/", '.env'))

    def set_env(self):

        self.MAIN_PATH = environ.get('MAIN_PATH')
        self.TELEGRAM_BINARY = environ.get("TELEGRAM_BINARY")
        self.LANGDETECT_MODEL = environ.get("LANGDETECT_MODEL")
        self.CHROMEDRIVER = environ.get("CHROMEDRIVER")
        self.TWITTER_AUTH = environ.get("TWITTER_AUTH")
        self.TWITTER2_AUTH = environ.get("TWITTER2_AUTH")
        self.REDDIT_AUTH = environ.get("REDDIT_AUTH")
        self.CT_AUTH = environ.get("CT_AUTH")
        self.CT_APP_AUTH = environ.get("CT_APP_AUTH")
        self.CT_INSTA_AUTH = environ.get("CT_INSTA_AUTH")
        self.FACEBOOK_AUTH = environ.get("FACEBOOK_AUTH")
        self.TELEGRAM_AUTH = environ.get("TELEGRAM_AUTH")
        self.GOOGLE_AUTH = environ.get("GOOGLE_AUTH")
        self.VKONTAKTE_AUTH = environ.get("VKONTAKTE_AUTH")
        self.YOUTUBE_AUTH = environ.get("YOUTUBE_AUTH")
        self.ALEXA_AUTH = environ.get("ALEXA_AUTH")
        self.MJ_AUTH = environ.get("MJ_AUTH")
        self.GAB_AUTH = environ.get("GAB_AUTH")
        self.FACEBOOK_BROWSER_AUTH = environ.get("FACEBOOK_BROWSER_AUTH")

    def get_auth(self):

        auths = {}
        with open(self.TWITTER_AUTH) as jloaded:
            auths["twitter"]=json.load(jloaded)
        with open(self.TWITTER2_AUTH) as jloaded:
            auths["twitter2"]=json.load(jloaded)
        with open(self.CT_AUTH) as jloaded:
            auths["crowdtangle"]=json.load(jloaded)
        with open(self.FACEBOOK_AUTH) as jloaded:
            auths["facebook"]=json.load(jloaded)
        with open(self.TELEGRAM_AUTH) as jloaded:
            auths["telegram"]=json.load(jloaded)
        with open(self.REDDIT_AUTH) as jloaded:
            auths["reddit"]=json.load(jloaded)
        with open(self.CT_APP_AUTH) as jloaded:
            auths["crowdtangle_app"]=json.load(jloaded)
        with open(self.CT_INSTA_AUTH) as jloaded:
            auths["crowdtangle_insta"]=json.load(jloaded)
        with open(self.GOOGLE_AUTH) as jloaded:
            auths["google"]=json.load(jloaded)
        with open(self.FACEBOOK_BROWSER_AUTH) as jloaded:
            auths["facebook_browser"]=json.load(jloaded)
        with open(self.VKONTAKTE_AUTH) as jloaded:
            auths["vkontakte"]=json.load(jloaded)
        with open(self.YOUTUBE_AUTH) as jloaded:
            auths["youtube"]=json.load(jloaded)
        with open(self.ALEXA_AUTH) as jloaded:
            auths["alexa"]=json.load(jloaded)
        with open(self.MJ_AUTH) as jloaded:
            auths["majestic"]=json.load(jloaded)
        with open(self.GAB_AUTH) as jloaded:
            auths["gab"]=json.load(jloaded)
        return auths
