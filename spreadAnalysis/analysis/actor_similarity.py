import pandas as pd
from datetime import datetime
import numpy as np
import nltk
import os
import pickle
import operator
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from spreadAnalysis.utils.link_utils import LinkCleaner
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import seaborn as sns
import scipy
import scipy.spatial
import sys

def to_default_datetime_format(date_str):

    return datetime.strptime(date_str[:19],"%Y-%m-%d %H:%M:%S")

def create_actor_vocab(awords,per_actor=250,min_occur=3):

    actor_vocab = set({})
    da_stopwords = set(stopwords.words('danish'))
    en_stopwords = set(stopwords.words('english'))
    sw_stopwords = set(stopwords.words('swedish'))
    for actor,wcounts in awords.items():
        pacount = 0
        for w,s in sorted(wcounts.items(), key=operator.itemgetter(1), reverse=True):
            if s >= min_occur:
                if "\n" not in w and len(w.replace(".","").replace(".","")) > 3:
                    if w not in da_stopwords and w not in en_stopwords and w not in sw_stopwords:
                        pacount+=1
                        actor_vocab.add(w)
            if pacount >= per_actor:
                #print (len(actor_vocab))
                break

    return actor_vocab

def create_actor_links(df,main_path,mode="sim_test",key_col="actor_name"):

    all_domains = defaultdict(int)
    actor_domains = {}
    link_path = "{0}/{1}.alinks".format(main_path,mode)
    if os.path.isfile(link_path):
        actor_domains,all_domains = pickle.load(open(link_path,"rb"))
    else:
        for i,row in df.iterrows():
            if row[key_col] not in actor_domains: actor_domains[row[key_col]]={}
            for url in set(list(LinkCleaner().get_url_list_from_text(str(row["text"])+" "+str(row["url"])))):
                domain = LinkCleaner().extract_domain(url)
                if domain is not None and len(domain) > 2:
                    if domain not in actor_domains[row[key_col]]: actor_domains[row[key_col]][domain]=0
                    actor_domains[row[key_col]][domain]+=1
                    all_domains[domain]+=1
        pickle.dump((actor_domains,all_domains),open(link_path,"wb"))

    return actor_domains, all_domains

def create_actor_post_times(df,main_path,mode="sim_test",key_col="actor_name",within_min=1):

    df = df.sort_values(['url', 'datetime'], ascending=[False, True], inplace=False)
    prev_url = None
    prev_ptime = None
    times_shared = None
    url_share_sets = {}
    first_in_streak = None
    for i,row in df.iterrows():
        #if row["url"] is None and row["url"] != np.nan: continue
        url = str(row["url"])
        ptime = to_default_datetime_format(str(row["datetime"]))
        actor = row[key_col]
        post_url = row["post_url"]
        if url == "nan": continue
        if prev_url is None or url != prev_url:
            prev_post_url = post_url
            prev_url = url
            prev_ptime = ptime
            prev_actor = actor
            times_shared = 0
            first_in_streak = prev_ptime
            url_share_sets[url+"_"+str(times_shared)]=set([])
        else:
            btw_share = ptime - prev_ptime
            btw_first_streak_share = ptime - first_in_streak
            if btw_share.total_seconds() < within_min*60 and btw_first_streak_share.total_seconds() < within_min*60:
                if not url+"_"+str(times_shared) in url_share_sets: url_share_sets[url+"_"+str(times_shared)]=set([])
                url_share_sets[url+"_"+str(times_shared)].add(prev_actor)
                url_share_sets[url+"_"+str(times_shared)].add(actor)
                #print (post_url)
                #print (prev_post_url)
                #sys.exit()
            else:
                times_shared+=1
                first_in_streak = ptime
        prev_url = url
        prev_ptime = ptime
        prev_actor = actor
        prev_post_url = post_url

    """for actor, s in url_share_sets.items():
        if len(s) > 0:
            print (actor)
            print (s)
    sys.exit()"""

    actor_post_times = {}
    for url, s in url_share_sets.items():
        share_comb = str(s)
        if not share_comb in actor_post_times:
            actor_post_times[share_comb]={}
        for actor in s:
            if not actor in actor_post_times[share_comb]:
                actor_post_times[share_comb][actor]=0
            actor_post_times[share_comb][actor]+=1

    print (len(actor_post_times))

    return actor_post_times

def create_word_count_data(df,main_path,mode="sim_test",include=["tfidf","word2vec","awords"],key_col="actor_name",new=True):

    idf_weights, awords, w2v_model = None, None, None
    tfidf_path = "{0}/{1}.tfidf".format(main_path,mode)
    awords_path = "{0}/{1}.awords".format(main_path,mode)
    word2vec_path = "{0}/{1}.model".format(main_path,mode)
    if os.path.isfile(tfidf_path):
        idf_weights = pickle.load(open(tfidf_path,"rb"))
    if os.path.isfile(awords_path):
        awords = pickle.load(open(awords_path,"rb"))
    if os.path.isfile(word2vec_path):
        w2v_model = Word2Vec.load(word2vec_path)
    if new:
        toker = nltk.tokenize.TweetTokenizer(preserve_case=False,reduce_len=True)
        documents = {}
        documents_counts = {}
        vocab_count = {}
        for i,row in df.iterrows():
            page = row[key_col]
            if page not in documents: documents[page]=[]
            if page not in documents_counts: documents_counts[page]=defaultdict(float)
            for token in toker.tokenize(str(row["text"])):
                if not token.isdecimal():
                    if len(token) > 2:
                        token = token.lower()
                        documents[page].append(token)
                        documents_counts[page][token]+=1.0
                        if token not in vocab_count: vocab_count[token]=0
                        vocab_count[token]+=1
        trans_documents = [list(v) for k,v in documents.items()]

        if "tfidf" in include:
            if idf_weights is None:
                tfidf_dict = TfidfVectorizer(analyzer=lambda x: x)
                tfidf_dict.fit(np.array(trans_documents))
                idf_weights = {w:tfidf_dict.idf_[i] for w, i in tfidf_dict.vocabulary_.items() if vocab_count[w]>1}
                pickle.dump(idf_weights,open(tfidf_path,"wb"))

        if "awords" in include:
            awords = {}
            for page,wcounts in documents_counts.items():
                if page not in awords: awords[page]={}
                for w,c in wcounts.items():
                    if w in idf_weights:
                        awords[page][w]=c*idf_weights[w]
            pickle.dump(awords,open(awords_path,"wb"))

        if "word2vec" in include:
            if w2v_model is None:
                w2v_model = Word2Vec(trans_documents, vector_size=150, window=12, min_count=3, workers=10)
                w2v_model.train(trans_documents, total_examples=len(trans_documents), epochs=15)
                w2v_model.save(word2vec_path)

    #for w,s in sorted(idf_weights.items(), key=operator.itemgetter(1), reverse=False):
        #print (w+" : "+str(s))

    return idf_weights, awords, w2v_model

def to_post_time_sim(actor,actor_post_times):

    vals = []
    cols = []

    """for url,s in sorted(actor_post_times.items()):
        cols.append("SAME_TIME_"+str(url))
        if actor in s:
            vals.append(1)
        else:
            vals.append(0)"""

    comb_count = 0
    for comb, actors in actor_post_times.items():
        cols.append("SAME_TIME_"+str(comb_count))
        if actor in actors:
            vals.append(actors[actor])
        else:
            vals.append(0)
        comb_count+=1

    new_df = pd.DataFrame(np.array([vals]), columns=cols)

    return new_df

def to_hour_of_day(df):

    hour_of_day = {r:0.0 for r in range(24)}
    for i,row in df.iterrows():
        h = to_default_datetime_format(row["datetime"]).hour
        hour_of_day[h]+=1.0

    hour_of_day = [v for k,v in sorted(hour_of_day.items(), key=lambda x: x[0])]
    mean_hour_of_day = np.mean(np.array(hour_of_day))
    std_hour_of_day = np.std(np.array(hour_of_day))
    new_cols = ["hour_"+str(r) for r in range(len(hour_of_day))]
    hour_of_day.append(mean_hour_of_day)
    hour_of_day.append(std_hour_of_day)
    new_cols.append("mean_hour_of_day")
    new_cols.append("std_hour_of_day")
    new_df = pd.DataFrame(np.array([hour_of_day]), columns=new_cols)

    return new_df

def to_indicators(df):

    indicators = []
    indicator_names = ["interactions","Russian_%","Norge_%"]
    page_likes = int(np.amax(np.array(list(df["followers"]))))
    cols = []
    for n in indicator_names:
        for n2 in indicator_names:
            if n != n2:
                indicators.append(np.mean(np.array([row[n]/row[n2] for i,row in df.iterrows() if row[n] is not None and row[n2] is not None and row[n2] > 0])))
                cols.append("{0}/{1}".format(n,n2))

    for n in indicator_names:
        indicators.append(np.mean(np.array([row[n] for i,row in df.iterrows() if row[n] is not None]))/float(len(df)))
        cols.append("{0}/n_posts".format(n))

    for n in indicator_names:
        indicators.append(np.mean(np.array([row[n] for i,row in df.iterrows() if row[n] is not None]))/float(page_likes))
        cols.append("{0}/followers".format(n))

    return pd.DataFrame(np.array([indicators]), columns=cols)

def to_word_vecs(actor_wcounts,vocab,idf_weights,w2v_model=None):

    actor_word_vec = []
    cols = []
    sum_w_count = float(np.sum(np.array(list(actor_wcounts.values()))))
    for w,wcount in sorted(actor_wcounts.items()):
        if w in vocab:
            actor_word_vec.append(wcount)
            cols.append("word_{}".format(w))

    if w2v_model is not None:
        actor_word2vec = []
        word2vec_cols = ["word2vec_{0}".format(r) for r in range(150)]
        actor_word2vec = np.mean([w2v_model.wv[w] * idf_weights[w] * (float(wcount/sum_w_count)) for w,wcount in sorted(actor_wcounts.items()) if w in w2v_model.wv] or [np.zeros(150)], axis=0)
        actor_word_vec.extend(actor_word2vec)
        cols.extend(word2vec_cols)

    return pd.DataFrame(np.array([actor_word_vec]), columns=cols)

def to_domains_shared(actor_dom_counts,all_domains,min_domain_count=3):

    cols = [domain for domain,c in sorted(all_domains.items()) if c >= min_domain_count]
    dom_vals = []
    for dom in cols:
        if dom in actor_dom_counts:
            dom_vals.append(actor_dom_counts[dom])
        else:
            dom_vals.append(0)

    return pd.DataFrame(np.array([dom_vals]), columns=cols)

def generate_data(main_path,outfile,input_data,min_per_actor=10):

    main_data = input_data
    main_data = main_data.replace({np.nan: None})
    key_col = "actor_name"
    print (len(main_data))
    if min_per_actor is not None:
        grouped = main_data.groupby(key_col)["message_id"].count()
        only_keys = set(grouped[grouped >= min_per_actor].index.tolist())
        main_data = main_data[main_data[key_col].isin(only_keys)]
    print (len(main_data))
    actor_post_times = create_actor_post_times(input_data,main_path)
    idf_weights, awords, w2v_model = create_word_count_data(main_data,main_path)
    actor_vocab = create_actor_vocab(awords)
    actor_domains, all_domains = create_actor_links(main_data,main_path)
    funcs = [to_post_time_sim]
    funcs.append(to_word_vecs)
    funcs.append(to_domains_shared)
    funcs.append(to_hour_of_day)
    funcs.append(to_indicators)
    actor_data = pd.DataFrame()

    for actor in list(main_data[key_col].unique()):
        print (actor)
        new_row = pd.DataFrame(np.array([actor]), columns=[key_col])
        one_actor = main_data[main_data[key_col]==actor]
        for func in funcs:
            if "to_word_vecs" in str(func):
                new_row = pd.concat([new_row,func(awords[actor],actor_vocab,idf_weights,w2v_model=w2v_model)], axis=1)
            elif "to_domains_shared" in str(func):
                new_row = pd.concat([new_row,func(actor_domains[actor],all_domains)], axis=1)
            elif "to_post_time_sim" in str(func):
                new_row = pd.concat([new_row,func(actor,actor_post_times)], axis=1)
            else:
                new_row = pd.concat([new_row,func(one_actor)], axis=1)
        actor_data = pd.concat([actor_data,new_row], axis=0)
    actor_data.to_csv(outfile)

    print (actor_data)
    print (actor_data.columns)

def get_mean_actor_vec(special_idx,df):

    special_df_vec = np.empty((1,df.shape[1]))
    df = StandardScaler().fit_transform(df)
    for idx in special_idx:
        special_df_vec = np.concatenate([special_df_vec,df[[idx]]], axis=0)
    special_df_vec = np.delete(special_df_vec, (0), axis=0)
    special_df_vec = special_df_vec.mean(0)
    special_df_vec = pd.DataFrame(special_df_vec).transpose()

    return special_df_vec

def find_similar_vector(special_idx,all_pages,df):

    special_df_vec = get_mean_actor_vec(special_idx,df)

    df = pd.DataFrame(df)
    test_df = df.copy()
    mins = []
    min_index = np.amax(np.array([scipy.spatial.distance.cdist(test_df, special_df_vec, metric='euclidean').min()**-1 for r in range(len(test_df)+len(special_idx))]))
    for r in range(len(df)+len(special_idx)):
        ary = scipy.spatial.distance.cdist(df, special_df_vec, metric='euclidean')
        idx = df[ary==ary.min()].index[0]
        print (all_pages[int(idx)] + ":" + str(((ary.min()**-1)/min_index)*100))
        df.drop(idx,inplace=True)

def compare_actor_groups(group1,group2,feature_names):

    compared = np.multiply(group1,group2)
    best_features = {}
    for i,val in enumerate(list(np.array(compared).flatten())):
        best_features[i]=val
    for i,v in sorted(best_features.items(), key=operator.itemgetter(1), reverse=True)[:100]:
        print (feature_names[i] + " : " + str(v))

def analyze_data(main_path,filename,must_share=False):

    key_col = "actor_name"
    special_pages = ["Khazarmaffian","Gula Västarna 2","Den Nakna Sanningen"]
    other_pages = ["Maskulint Initiativ","Alternativ för Sverige - AfS Göteborg","NEJ till gratis körkort och alla andra särrättigheter för invandrare"]
    main_data = pd.read_csv(main_path+filename)
    main_data = main_data.replace({np.nan: 0.0})

    if must_share:
        only_shared_positives = set([])
        for col in main_data.columns:
            if col != key_col:
                col_check = []
                for page in special_pages:
                    col_check.append(list(main_data[main_data[key_col]==page][col])[0])
                col_check = [i for i in col_check if i > 0 or i < 0]
                if int(len(col_check)) > int(len(special_pages)*0.5):
                    only_shared_positives.add(col)
        print (len(only_shared_positives))
    else:
        only_shared_positives = set(main_data.columns)

    pages = main_data.pop(key_col)
    X = main_data
    Y = [1 if p in special_pages else 0 for p in pages]
    Yo = [1 if p in other_pages else 0 for p in pages]
    #find_similar_vector([i for i,e in enumerate(Y) if e == 1],pages,X)
    compare_actor_groups(get_mean_actor_vec([i for i,e in enumerate(Y) if e == 1],X),get_mean_actor_vec([i for i,e in enumerate(Yo) if e == 1],X),X.columns)
    sys.exit()

    important_features = {}
    feature_names = X.columns
    for r in range(150):
        np.random.seed(r)
        X = StandardScaler().fit_transform(X)
        model = ExtraTreesClassifier(n_estimators=30)
        model.fit(X, Y)
        these_important_features = {n:f for f,n in zip(model.feature_importances_,feature_names) if f > 0}
        for f,n in these_important_features.items():
            if f in important_features:
                important_features[f]+=n
            else:
                important_features[f]=n

    list_count = 0
    for f,n in sorted(important_features.items(), key=operator.itemgetter(1), reverse=True):
        if f in only_shared_positives and "word2vec" not in f:
            list_count+=1
            print (str(f)+"\t"+str(n))
        if list_count > 100:
            break

def clustering(main_path,filename,key_col="actor_name"):

    main_data = pd.read_csv(filename)
    main_data = main_data.replace({np.nan: 0.0})
    pages = main_data.pop(key_col)
    X = main_data
    clusterer = hdbscan.HDBSCAN(**{'min_cluster_size':2,"min_samples":2}).fit(X)
    labels = clusterer.labels_
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    projection = TSNE().fit_transform(X)
    plt.scatter(*projection.T, c=colors)
    #plt.scatter(np.random.random((7,5)),np.random.random((7,5)))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.legend(handles=[mpatches.Patch(color=palette[x], label=x) for x in set(labels) if x >= 0])
    plt.title('Clusters found by {}'.format(str("HDBSCAN")), fontsize=24)
    plt.savefig("{0}/P_clusters.png".format(main_path))

    actor_labels = {}
    #print ("actor" + "\t" + "cluster")
    for actor, label in sorted(zip(pages,labels), key=operator.itemgetter(1), reverse=True):
        print (actor + "\t" + str(label))
        if label not in actor_labels: actor_labels[label]=[]
        actor_labels[label].append(actor)


if __name__ == "__main__":
    mode = sys.argv[1]
    main_path = sys.argv[2]
    #mode = "gen2_no"
    #main_path = "/Users/jakobbk/Documents/postdoc/p15/gen2_no"
    posts_file = main_path+"/"+mode+"_posts.csv"
    out_filename = main_path+"/"+mode+"_actor_sim.csv"
    input_data = pd.read_csv(posts_file)
    if sys.argv[3] == "make":
        generate_data(main_path,out_filename,input_data)
    if sys.argv[3] == "cluster":
        clustering(main_path,out_filename)
    if sys.argv[3] == "sim":
        analyze_data(main_path,out_filename)
