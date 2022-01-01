from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.io.config_io import Config

conf = Config()
#ct = Crowdtangle(conf.get_auth()["crowdtangle"])
ct = Crowdtangle(conf.get_auth()["crowdtangle"])

#url = "https://nyheteridag.se/thunbergs-klimatrorelse-tar-stallning-mot-israel-apartheidstat/"
#url = "https://www.nyatider.nu/frankrikes-toppgeneral-varnar-vi-far-inte-delta-i-en-konfrontation-mellan-usa-och-kina/"
#url = "https://www.facebook.com/Politiskt.Inkorrekt/videos/2204008066498451/"
#url = "https://nyheteridag.se/ranarna-avslutade-med-att-urinera-pa-offret-de-garvar-och-sager-val-javla-vidriga-svenne"
#url = "http://katerinamagasin.se/medan-allt-fler-svenska-hemlosa-svalter-fortsatter-sverige-betala-miljoner-till-kriminella-afghaner/"
#ref_data = ct.url_referals(url,start_date="2010-01-01")
#actor = "https://www.facebook.com/cphinfocenter.2014"
#actor = "https://www.facebook.com/infodirekt/?fref=ts"
#actor = "https://www.facebook.com/newsfront99/"
#actor = "https://www.facebook.com/groups/1357640650938923/permalink/"
#actor = "friatider"
#actor = "2476461525767081"
#actor = "250968892087993"
#actor = "Svenskazoner"
#actor = "https://www.facebook.com/1066376647150124"
#actor = "https://www.facebook.com/steenchristiansen37"
actor = "https://www.instagram.com/steen2620"
actor = "dkdox.tv"
query = '"danske værdier"'
#data = ct.actor_content(actor,start_date="2021-08-01")

data = ct.query_referals(query,start_date="2021-12-12")
print (data)
print (len(data["output"]))
