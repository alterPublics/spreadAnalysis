from spreadAnalysis.reqs.req import Req
from spreadAnalysis.utils.link_utils import LinkCleaner
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from datetime import datetime
import random
import pandas as pd

class Youtube:

    def __init__(self,tokens):

        self.tokens = tokens["tokens"]
        self.base_url = "https://www.googleapis.com/youtube/v3"

        self.actor_info = {}

    def _update_output(self,res,output_data):

        if res is not None and res.ok:
            for e in res.json()["items"]:
                if e["snippet"]["channelId"] in self.actor_info:
                    e["actor"]=self.actor_info[e["snippet"]["channelId"]]
                output_data["output"].append(e)
        return output_data

    def _get_next_token(self,res):
        next_token = None
        if res is not None and res.ok:
            res = res.json()
            if "nextPageToken" in res:
                next_token = res["nextPageToken"]
        return next_token

    def _get_upload_playlist_from_channel(self,actor):

        upload_list = []
        call_url = self.base_url+"/channels"
        params = {  "part":"snippet,contentDetails,statistics",
                    "key":random.choice(self.tokens)["key"]}
        if "youtube." in actor and ("channel/" in actor or "/c/" in actor):
            params["id"] = LinkCleaner().extract_username(actor,never_none=True)
        elif "youtube." in actor:
            params["forUsername"] = LinkCleaner().extract_username(actor,never_none=True)
        if ("forUsername" in params and len(params["forUsername"]) < 1) or ("id" in params and len(params["id"]) < 1):
            params["id"]=actor
        if "id" not in params and "forUsername" not in params:
            params["id"]=actor
        res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=0)
        if res is not None and res.ok:
            if "items" in res.json():
                selected_channel = [item for item in res.json()["items"]][0]
                upload_list = selected_channel["contentDetails"]["relatedPlaylists"]["uploads"]
                self.actor_info[selected_channel["id"]]=selected_channel
            else:
                print ("NO ITEMS")
                print (res.json())
        else:
            upload_list = None

        return upload_list

    def _get_video_ids_from_playlist(self,playlist,max_results=35000):

        video_ids = []
        call_url = self.base_url+"/playlistItems"
        params = {  "part":"snippet,contentDetails",
                    "key":random.choice(self.tokens)["key"],
                    "playlistId":"{0}".format(playlist),
                    "maxResults":50 }
        res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=0)
        if res is not None and res.ok:
            video_ids.extend([vid["contentDetails"]["videoId"] for vid in res.json()["items"]])
            next_token = self._get_next_token(res)
            params["pageToken"]=next_token
            while next_token:
                res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=0)
                video_ids.extend([vid["contentDetails"]["videoId"] for vid in res.json()["items"]])
                next_token = self._get_next_token(res)
                params["pageToken"]=next_token
                if len(video_ids) > max_results:
                    break

        return video_ids

    def _get_videos_from_ids(self,video_ids,start_date=None,end_date=None):

        video_data = {"output":[]}
        call_url = self.base_url+"/videos"
        for ids in hlp.chunks(video_ids,50):
            params = {  "part":"snippet,statistics,contentDetails",
                        "key":random.choice(self.tokens)["key"],
                        "id":",".join(ids)}
            res = Req.get_response(call_url,params=params,fail_wait_time=40,wait_time=0)
            self._update_output(res,video_data)
            #if not hlp.date_is_between_dates(Spread._get_date(video_data["output"][-1]),start_date,end_date):

        return video_data

    def _get_data(self,data,input_type,endpoint,wait_time=0,start_date=None,end_date=None):

        if input_type == "actor":
            upload_list = self._get_upload_playlist_from_channel(endpoint)
            if upload_list is not None:
                video_ids = self._get_video_ids_from_playlist(upload_list)
                video_data = self._get_videos_from_ids(video_ids,start_date=start_date,end_date=end_date)
                data["output"]=video_data["output"]
            else:
                data = None

        return data

    def actor_content(self,actor,start_date=None,end_date=None):

        start_date, end_date = hlp.get_default_dates(start_date,end_date)
        data = {"input":actor,
                "input_type":"actor",
                "output":[],
                "method":"youtube"}

        return self._get_data(data,data["input_type"],actor,start_date=start_date,end_date=end_date)
