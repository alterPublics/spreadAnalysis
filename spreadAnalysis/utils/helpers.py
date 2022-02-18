from datetime import datetime
from datetime import timedelta
import random
import time
from langdetect import detect_langs
import fasttext
from itertools import islice
from operator import itemgetter

def to_default_date_format(date_str):

    if isinstance(date_str,str):
        return datetime.strptime(date_str[:10],"%Y-%m-%d")
    else:
        return date_str

def get_default_dates(start_date,end_date):

    if start_date is None:
        start_date = f"{str(datetime.now().year)}-01-01"
    if end_date is None:
        end_date = str(datetime.now())[:10]

    return start_date,end_date

def create_date_ranges(start_date,end_date,interval=2,overlap=0):

    if isinstance(start_date,datetime):
        since_date = start_date
    else:
        since_date = datetime.strptime(start_date,"%Y-%m-%d")
    if isinstance(end_date,datetime):
        until_date = end_date
    else:
        until_date = datetime.strptime(end_date,"%Y-%m-%d")

    date_ranges = []
    current_until_date = since_date+timedelta(days=interval)
    current_since_date = since_date
    while current_until_date < until_date:
        date_ranges.append((current_since_date,current_until_date-timedelta(days=1)+timedelta(days=1)))
        current_since_date = current_since_date+timedelta(days=interval)-timedelta(days=overlap)
        current_until_date = current_until_date+timedelta(days=interval)-timedelta(days=overlap)

    date_ranges.append((current_since_date,until_date))
    return date_ranges

def get_next_end_date(end_date,stop_date,interval=2,max_interval=365,check_start_date=None):

    if not isinstance(end_date,datetime):
        end_date = datetime.strptime(str(end_date)[:10],"%Y-%m-%d")
    if not isinstance(stop_date,datetime):
        stop_date = datetime.strptime(str(stop_date)[:10],"%Y-%m-%d")
    if interval < max_interval:
        next_end_date = end_date+timedelta(days=interval)
    else:
        next_end_date = end_date+timedelta(days=max_interval)
    if next_end_date > stop_date:
        next_end_date = stop_date
    if check_start_date is not None:
        if str(next_end_date)[:10] == str(check_start_date)[:10]:
            next_end_date = next_end_date+timedelta(days=interval+2)
    return next_end_date

def get_diff_in_days(start_date,end_date):

    diff = to_default_date_format(end_date)-to_default_date_format(start_date)
    return diff.days

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunks_dict(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def chunks_optimized(data, n_chunks=2):

    chunked = {n:{} for n in range(n_chunks)}
    chunks_allo = {n:0 for n in range(n_chunks)}
    for k,v in data.items():
        lowest = sorted(chunks_allo.items(), key=itemgetter(1), reverse=False)[0][0]
        chunked[lowest][k]=v
        chunks_allo[lowest]+=len(v)
    print (chunks_allo)
    return list([v for v in chunked.values()])

def random_wait(between=(1,3),skip_wait=None):

    if skip_wait is not None:
        if float(random.random()) < skip_wait:
            return False
    thoughtful_human = 0.0
    if random.randint(1,100) > 91:
        thoughtful_human = float(random.randint(1,88))
    secs = float(random.randint(between[0],int(between[1]-1))+round(float(random.random()),3))+thoughtful_human
    time.sleep(secs)

def get_lang_and_conf(text,min_conf=.75,model=None):

    lang = "und"
    lang_conf = 0.0

    if model is not None:
        text = text.replace('\n','')
        predictions = model.predict(text, k=3)
        conf = float(predictions[1][0])
        if conf >= min_conf:
            lang = str(predictions[0][0]).replace("__label__","")
            lang_conf = conf
    else:
        if len(text) > 0:
            try:
                suggestions = detect_langs(text)
            except Exception as e:
                suggestions = []
                #print (e)
            if len(suggestions) > 0:
                lang = str(suggestions[0]).split(":")[0]
                lang_conf = float(str(suggestions[0]).split(":")[1])
                if float(lang_conf) < .75:
                    lang = "und"

    return {"lang":lang,"lang_conf":lang_conf}

def date_is_between_dates(_date,start_date,end_date):

    if isinstance(start_date,str):
        start_date = to_default_date_format(start_date)
    if isinstance(end_date,str):
        end_date = to_default_date_format(end_date)
    if isinstance(_date,str):
        _date = to_default_date_format(_date)
    _date = _date.replace(tzinfo=None)

    if _date >= start_date and _date <= end_date:
        return True
    else:
        return False
