from datetime import datetime
from datetime import timedelta
import inspect
import hashlib

class StructUtils:

    def as_dict(self):

        atts = [a[0] for a in inspect.getmembers(self.__class__,
                            lambda a:not(inspect.isroutine(a))) if "__" not in a[0]]
        data_dict = {a:getattr(self,a) for a in atts}
        if self.__id__ is not None:
            data_dict["id"]=self.__id__

        return data_dict

    @classmethod
    def utime_to_datetime(cls,utime):
        try:
            return datetime.fromtimestamp(int(utime))
        except:
            #print ("input not UNIX time integer. \nRecieved {0}. \nReturning None.".format(utime))
            return None

    @classmethod
    def number_text_to_integer(cls,number_text,default=0):
        count = default
        number_text = str(number_text)
        if "," in number_text:
            count = int(number_text.replace(",","")+"00")
        elif "tusind" in number_text:
            count = int(number_text+"000")
        else:
            try:
                count = int(number_text)
            except Exception as e:
                count = 0
                print (e)

        return count

    @classmethod
    def _safe_get_attr(cls,data,idens=[]):

        attr = None
        if len(idens) > 1:
            for iden in idens:
                new_data = cls._safe_get_attr(data,idens=[idens.pop(0)])
                if new_data is not None:
                    attr = cls._safe_get_attr(new_data,idens=idens)
        elif len(idens) < 1:
            pass
        else:
            try:
                attr = data[idens[0]]
            except:
                pass
        return attr

    @classmethod
    def determine_emoji(cls,class_text):

        emoji = None
        if class_text == "_59aq img sp_FnpQQMXWL5W sx_4760b1":
            emoji = "LIKE"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_e0ad16":
            emoji = "ANGRY"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_5a6c86":
            emoji = "SAD"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_b2274b":
            emoji = "LOVE"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_2abdad":
            emoji = "HAHA"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_019a3f":
            emoji = "WOW"
        elif class_text == "_59aq img sp_FnpQQMXWL5W sx_6477b5":
            emoji = "CARE"

        return emoji

    @classmethod
    def month_da_to_month_en(cls,month):

        en_month = month
        if month == "oktober": en_month = "october"
        elif month == "januar": en_month = "january"
        elif month == "februar": en_month = "february"
        elif month == "marts": en_month = "march"
        elif month == "maj": en_month = "may"
        elif month == "juni": en_month = "june"
        elif month == "juli": en_month = "july"
        return en_month

    @classmethod
    def text_field_to_datetime(cls,text_field):

        #print (text_field)
        date_object = None
        if "  ·" in text_field or "  · " in text_field:
            text_field = text_field.split("  · ")[1].split("  ·")[0].strip()
        text_field = str(text_field).lower()
        if text_field[-2:] == "t.":
            text_field = text_field.split(" t.")[0]
            hour = str((datetime.now()- timedelta(hours=int(text_field))).hour)
            day = str((datetime.now()- timedelta(hours=int(text_field))).day)
            month = str(datetime.now().month)
            year = str(datetime.now().year)
            if len(day) < 2: day = "0"+day
            date_object = datetime.strptime(f'{day} {month} {year} {hour}', "%d %m %Y %H")
        elif "timer" in text_field:
            text_field = text_field.split(" timer")[0]
            hour = str((datetime.now()- timedelta(hours=int(text_field))).hour)
            day = str((datetime.now()- timedelta(hours=int(text_field))).day)
            month = str(datetime.now().month)
            year = str(datetime.now().year)
            if len(day) < 2: day = "0"+day
            date_object = datetime.strptime(f'{day} {month} {year} {hour}', "%d %m %Y %H")
        elif "i går" in text_field:
            minute = text_field.split(".")[-1].strip()
            hour = text_field.split("kl. ")[-1].split(".")[0]
            day = str((datetime.now()- timedelta(days=1)).day)
            month = str(datetime.now().month)
            year = str(datetime.now().year)
            if len(day) < 2: day = "0"+day
            date_object = datetime.strptime(f'{day} {month} {year} {hour} {minute}', "%d %m %Y %H %M")
        elif "kl." not in text_field:
            day = text_field.split(".")[0].split(" ")[-1]
            month = cls.month_da_to_month_en(text_field.split(". ")[1].split(" ")[0]).capitalize()
            year = text_field.split(" ")[-1].replace(".","")
            date_object = datetime.strptime(f'{day} {month} {year}', "%d %B %Y")
        elif len(text_field.split(" ")) > 2 and len(text_field.split(" ")[2])<4:
            #print ("111111")
            minute = text_field.split(".")[-1].strip()
            hour = text_field.split("kl. ")[-1].split(".")[0]
            day = text_field.split(".")[0].split(" ")[-1]
            month = cls.month_da_to_month_en(text_field.split(". ")[1].split(" ")[0]).capitalize()
            year = str(datetime.now().year)
            if len(day) < 2: day = "0"+day
            date_object = datetime.strptime(f'{day} {month} {year} {hour} {minute}', "%d %B %Y %H %M")
        else:
            #print ("222222")
            minute = text_field.split(".")[-1].strip()
            hour = text_field.split("kl. ")[-1].split(".")[0]
            day = text_field.split(".")[0].split(" ")[-1]
            month = cls.month_da_to_month_en(text_field.split(". ")[1].split(" ")[0]).capitalize()
            year = text_field.split(" kl.")[0].split(" ")[-1]
            if len(day) < 2: day = "0"+day
            date_object = datetime.strptime(f'{day} {month} {year} {hour} {minute}', "%d %B %Y %H %M")

        return date_object

    @classmethod
    def hash_and_salt_text(cls,text):
        #use length of text and chars in text for salt
        #print (text)
        text_length = len(text)
        letters = text[0:text_length:2]
        salt = bytes(text_length) + str.encode(letters)
        hash_bytes = hashlib.pbkdf2_hmac('sha256', text.encode('utf-8'), salt, 100000)

        return str(hash_bytes)
