from spreadAnalysis.export.export import actor_export_default, domain_shares_per_actor,domain_shares_per_com
from spreadAnalysis.export.export import combine_exports, cluster_analysis
from spreadAnalysis.export.export import actor_net_export_default,actor_export_default_corrected
from spreadAnalysis.export.export import actor_export_project_default, link_full_text_export
from spreadAnalysis.export.export_schemas import actor_export_russian_dk,posts_export_russian_dk
from spreadAnalysis.export.export_schemas import explore_actor_net, explore_actor_net_content
from spreadAnalysis.export.export_schemas import explore_domain_profiles
from spreadAnalysis.export.export_schemas import export_net_slice
import sys
import pandas as pd

if __name__ == "__main__":
	args = sys.argv
	if args[1] == "all":
		#domain_shares_per_com("/home/alterpublics/projects/altmed",com_var="inner_com")
		#explore_domain_profiles("/home/alterpublics/projects/altmed",com_var="inner_com")
		#sys.exit()
		all_project_titles = ["altmed_denmark","altmed_germany","altmed_sweden","altmed_austria"]
		all_paths = ["/home/alterpublics/projects/altmed_denmark",
					"/home/alterpublics/projects/altmed_germany",
					"/home/alterpublics/projects/altmed_sweden",
					"/home/alterpublics/projects/altmed_austria"]
		all_nets = ["alt_dk","alt_de","alt_sv","alt_at"]
		nat_langs=[["da"],["de","de-AT","de-DE"],["sv"],["de","de-AT","de-DE"]]
		for main_path,title,net_title,nat_lang in zip(all_paths,all_project_titles,all_nets,nat_langs):
			if "sweden" in main_path or "denmark" in main_path:
				#actor_net_export_default(main_path,title,net_title)
				actor_export_default(main_path,title,net_title,nat_langs=nat_lang,no_pop_case=False)
			#actor_export_default(main_path,title,net_title,nat_langs=nat_lang,no_pop_case=True)
		combine_exports(all_paths,all_project_titles,export_path="/home/alterpublics/projects/altmed/all_actors.csv")
		#explore_actor_net("/home/alterpublics/projects/altmed")
		#explore_actor_net_content("/home/alterpublics/projects/altmed",actor_p="PParzival_Twitter",url=None)
		#explore_actor_net_content("/home/alterpublics/projects/altmed",actor_p=None,url="https://dipbt.bundestag.de/dip21/btd/17/120/1712051.pdf")

	if args[1] == "de":
		main_path = "/home/alterpublics/projects/altmed_germany"

		#domain_shares_per_actor(main_path,"altmed_germany","alt_de")
		#sys.exit()
		#actor_export_default(main_path,"altmed_germany","alt_de")
		actor_export_default(main_path,"altmed_germany_small","alt_de_v2")
		actor_export_default(main_path,"altmed_denmark","alt_dk")
		actor_export_default(main_path,"altmed_sweden","alt_sv")
		actor_export_default(main_path,"altmed_austria","alt_at")

	if args[1] == "rt":
		main_path = "/home/alterpublics/projects/rt_sputnik"

		#actor_export_project_default(main_path,["rt_sputnik"],add_webs=["de.rt.com"],limit=10000000000)
		#df = pd.read_csv(main_path+"/{0}_post_export_{1}.csv".format("rt_sputnik",0))
		#print (df.head())
		link_full_text_export(main_path,"rt_sputnik",main_path+"/{0}_post_export_{1}.csv".format("rt_sputnik",0))
		actor_export_project_default(main_path,["rt_sputnik"],add_webs=["de.rt.com"],limit=10000000000)

	if args[1] == "rt_dk":
		main_path = "/home/alterpublics/projects/altmed_denmark"

		posts_export_russian_dk(main_path,"altmed_denmark","alt_dk")
		#actor_export_russian_dk(main_path,"altmed_denmark","alt_dk",nat_langs=["da"])

	if args[1] == "slice":
		main_path = "/home/alterpublics/projects/altmed"

		for title in ["altmed_denmark","altmed_sweden"]:
			export_net_slice(main_path,titles=[title])

	if args[1] == "corr":
		main_path = "/home/alterpublics/projects/altmed"

		actor_export_default_corrected(main_path)

	if args[1] == "cluster":
		main_path = "/home/alterpublics/projects/altmed"
		cluster_analysis(main_path)
