from spreadAnalysis.scraper.scraper import Scraper
from bs4 import BeautifulSoup
import datetime

class TwScraper(Scraper):

    def __init__(self,settings={}):
        Scraper.__init__(self,settings=settings)

        self.default_date_interval = 360

    def set_date_interval(self,interval):

        self.default_date_interval = interval

    def create_date_ranges(self,start_date,end_date,interval=None):

        if interval is None: interval = self.default_date_interval
        since_date = datetime.datetime.strptime(start_date,"%Y-%m-%d")
        until_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")

        date_ranges = []
        current_until_date = since_date+datetime.timedelta(days=interval)
        current_since_date = since_date
        while current_until_date < until_date:
            date_ranges.append((current_since_date,current_until_date-datetime.timedelta(days=1)+datetime.timedelta(days=1)))
            current_since_date = current_since_date+datetime.timedelta(days=interval)
            current_until_date = current_until_date+datetime.timedelta(days=interval)

        date_ranges.append((current_since_date,until_date))
        return date_ranges

    def is_no_results(self,url):

        self.browser.get(url)
        time.sleep(random.uniform(2.5,4.2))
        no_result = False
        html = self.browser.page_source
        soup = BeautifulSoup(str(html), self.default_soup_parser)
        results = soup.find_all("div",{"class":"css-901oao r-hkyrab r-1qd0xha r-1b6yd1w r-vw2c0b r-ad9z0x r-15d164r r-bcqeeo r-q4m81j r-qvutc0"})
        for box in results:
            if hasattr(box,"text"):
                if "No results for" in str(box.text):
                    no_result = True

        return no_result

    def collect_tweet_ids(self,query,end_date=None,start_date=None,interval=None,verbose=False):

        start_date, end_date = hlp.get_default_dates(start_date,end_date)

        tweets_ids_found = set([])
        date_ranges = self.create_date_ranges(start_date,end_date,interval=interval)
        url = "https://twitter.com/search?f=live&vertical=default&q={query}&src=typd".format(query=query)
        if self.is_no_results(url):
            return False
        for dr in date_ranges:
            sdate = str(dr[0])[:10]
            udate = str(dr[1])[:10]
            url = "https://twitter.com/search?f=live&vertical=default&q={query}%20since%3A{sdate}%20until%3A{udate}&src=typd".format(query=query,sdate=sdate,udate=udate)
            if verbose: print (url)
            self.browser_reset()
            self.browser.get(url)
            time.sleep(random.uniform(4.5,7.2))
            wait_ = random.uniform(1.7,1.7+2.1)
            lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            match=False
            while(match==False):
                time.sleep(wait_)
                lastCount = lenOfPage
                lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
                html = self.browser.page_source
                soup = BeautifulSoup(str(html), self.default_soup_parser)
                results = soup.find_all("div",{"class":"css-1dbjc4n r-1d09ksm r-18u37iz r-1wbh5a2"})
                for result in results:
                    if not "/status/" in str(result): continue
                    tweet_id = str(result).split("/status/")[1].split('"')[0]
                    tweets_ids_found.add(tweet_id)
                if lastCount==lenOfPage:
                    match=True

        if len(tweets_ids_found) > 500:
            print ("Found more than 500 matches. Recalling with smaller date interval...")
            if interval is None:
                interval = int(self.default_date_interval/2)
            else:
                interval = int(interval/2)
            tweets_ids_found = self.collect_tweet_ids(query,end_date=end_date,start_date=start_date,interval=interval)

        return tweets_ids_found
