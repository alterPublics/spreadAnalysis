from spreadAnalysis.scraper.fb_scraper import FbScraper

post_id = "1904850382982249"
fbs = FbScraper(settings={"cookie_path":"/Users/jakobbk/Downloads/fb_tests","exe_path":"/usr/local/bin/chromedriver","machine":"local"})
fbs.browser_init()
fbs.fb_login(user="jakob.kristensen@pg.canterbury.ac.nz",pwd="jegvilpaakokkeskole")
data = fbs.get_post_shares(post_id)
fbs.browser_quit()

for d in data:
    print (d.as_dict())
