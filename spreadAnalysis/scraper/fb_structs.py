from bs4 import BeautifulSoup
import json

from spreadAnalysis.utils.struct_utils import StructUtils

class PostShare(StructUtils):

    member_numeric_id = None
    member_name = None
    member_id = None
    post_id = None

    html = None

    def __init__(self,html,parser,post_id):

        self.html = html
        self.parser = parser
        self.post_id = post_id

        self.__id__ = None
        self.parse_html()

    def _set_member_info(self,soup):

        member_name = None
        member_id = None
        for name_box in soup.find_all("div",{"class":"_1uja _xon"}):
            for name_link in BeautifulSoup(str(name_box),self.parser).find_all("a"):
                try:
                    if "profile.php?" in str(name_link["href"]):
                        member_id = str(name_link["href"]).split("?id=")[-1].split("&")[0]
                    else:
                        member_id = str(name_link["href"]).split("?")[0].replace("/","")
                    break
                except:
                    pass
            member_name = name_box.text.split("Anmodning")[0]

        self.member_name = member_name
        self.member_id = member_id
        #print (member_id)
        # Hashing user IDs
        #self.member_id = self.hash_and_salt_text(str(self.member_id))
        #self.member_name = self.member_name

    def get_id(self):

        return self.__id__

    def as_dict(self):

        data = super(PostShare, self).as_dict()
        del data["html"]
        data["user_id"]=data["member_id"]
        data["name"]=data["member_name"]
        data["numeric_id"]=data["member_numeric_id"]
        return data

    def parse_html(self):

        reaction_soup = BeautifulSoup(str(self.html),self.parser)
        self._set_member_info(reaction_soup)
        self.__id__ = self.post_id+"_"+self.member_id
