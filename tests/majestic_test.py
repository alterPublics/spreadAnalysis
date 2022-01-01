from spreadAnalysis.some.majestic import Majestic
from spreadAnalysis.io.config_io import Config

conf = Config()
mj = Majestic(conf.get_auth()["majestic"])

#url = "https://nyheteridag.se/thunbergs-klimatrorelse-tar-stallning-mot-israel-apartheidstat/"
#url = "https://www.nyatider.nu/frankrikes-toppgeneral-varnar-vi-far-inte-delta-i-en-konfrontation-mellan-usa-och-kina/"
#url = "https://www.facebook.com/Politiskt.Inkorrekt/videos/2204008066498451/"
#url = "https://nyheteridag.se/ranarna-avslutade-med-att-urinera-pa-offret-de-garvar-och-sager-val-javla-vidriga-svenne"
#url = "http://katerinamagasin.se/medan-allt-fler-svenska-hemlosa-svalter-fortsatter-sverige-betala-miljoner-till-kriminella-afghaner/"
#ref_data = ct.url_referals(url,start_date="2010-01-01")

url = "https://direktaktion.nu/debatt/2021/07/08/hemtjanst-eller-otjanst-all-makt-at-vardarbetarna/"
#url = "https://majestic.com/reports/site-explorer?folder=&q=https%3A%2F%2Fwww.s-sanningen.com%2Fbarnhemsbarnen&IndexDataSource=F"
url = "https://extremnews.com"
data = mj.domain_referals(url)
print (data)
print (len(data))
