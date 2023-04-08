#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import operator


# In[4]:


import math


# In[5]:


from pandas.api.types import is_datetime64_any_dtype as is_datetime


# In[6]:


from pandas.plotting import scatter_matrix


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


from sklearn.metrics.pairwise import linear_kernel


# In[9]:


import spacy


# In[10]:


import en_core_web_sm


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


from wordcloud import STOPWORDS, WordCloud


# In[13]:


wc = WordCloud()


# In[14]:


from nltk.corpus import stopwords


# In[15]:


import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


import numpy as np


# In[17]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


# In[18]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize,word_tokenize 


# In[19]:


from nltk.stem.snowball import SnowballStemmer


# In[20]:


snowball = SnowballStemmer('english')


# In[21]:


from nltk.stem import WordNetLemmatizer

wordnet_lem = WordNetLemmatizer()


# In[22]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# In[23]:


from nltk.corpus import wordnet


# In[24]:


from nltk import Text


# In[25]:


nlp = spacy.load("en_core_web_sm")


# In[26]:


import pandas as pd


# In[27]:


import pymysql


# In[28]:


import MySQLdb


# In[29]:


import sqlalchemy


# In[30]:


from sqlalchemy import create_engine


# In[31]:


from ast import literal_eval


# In[32]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


# In[33]:


from collections import Counter


# In[60]:


import pretty_html_table


# In[61]:


from pretty_html_table import build_table


# In[34]:


engine = create_engine('mysql://root:PostgrePython_098@localhost:3306/movie_db')


# In[35]:


con = engine.connect()


# In[36]:


def create_form(x):
    return ' '.join(x)


# In[ ]:





# In[ ]:





# ## Defining functions required for the recommendation

# In[37]:


def spacy_ent(text):
    return [(X.text, X.label_) for X in nlp(text).ents]


# In[38]:


def spacy_change(text, ent):
    #list_ent = ent.strip("][").split(", ")
    for tup in ent:
        if tup[1]=='PERSON':
            text = text.replace(tup[0], str.lower(tup[0].replace(" ","")))
        else:
            continue
        
    return text


# In[39]:


stop = stopwords.words('english')
def remove_stop(text):
    return [w for w in text if w not in stop]


# In[40]:


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[41]:


def tokenizer_snowball(text):
    return [snowball.stem(word) for word in text.split()]


# In[42]:


def lemmatizer(text):
    res = []
    
    for w in nltk.word_tokenize(text):
        res.append(wordnet_lem.lemmatize(w))
        
    return res


# In[43]:


def nouns_isolated(text):
    nouns = []
    
    for w in text:
        syns = wordnet.synsets(w)
        if syns and syns[0].lexname().split('.')[0]=='noun':
            nouns.append(w)

    return nouns


# In[44]:


def get_sentiment_pos(part):
    headers = ['pos','neg','neu','compound']
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(part)
    pos=compound=neu=neg=0
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        pos+=vs['pos']/(len(sentences))
        #compound+=vs['compound']/(len(sentences))
        #neu+=vs['neu']/(len(sentences))
        #neg+=vs['neg']/(len(sentences))
    return pos


# In[45]:


def create_again(x, title, num_times, movie_name):
    if title==movie_name:
        list_x = x.split(' ')
        new_vals = ''
        for item in list_x:
            if item in num_times.keys():
                new_vals += ' '+(item+' ')*num_times[item]
    else:
        return x
    return new_vals


# In[46]:


def movie_items_extend_keyws(x, more_keyws):
    #list_x = x.split(' ')
    x=x.tolist()[0]
    x.extend(more_keyws)
    res = [i for n, i in enumerate(x) if i not in x[:n]]

    return res


# In[47]:


def keywords_extend_keyws(x, title, movie_name, more_keyws):
    if title==movie_name:
        list_x = x.split(' ')
        list_x.extend(more_keyws)
        res = [i for n, i in enumerate(list_x) if i not in list_x[:n]]
        new_vals = ''
        for item in res:
                new_vals += ' '+item+' '
    else:
        return x
    return new_vals


# In[48]:


def tokenization(x):
    tokens = tokenizer.tokenize(x)
    reconstructed = ''
    
    for token in tokens:
        reconstructed = reconstructed + ' ' + token
    
    return reconstructed


# ## Recommendation Function

# In[101]:


def recommendation_engine(movie_name):
    
    df = pd.read_sql_table('movie_table',con)
    
    print(df.shape)
    
    df = df.drop(columns = ['pk_id'])
    
    print(df.shape)
    
    df = df.drop_duplicates()
    
    df.reset_index(drop=True, inplace=True)
    
    print(df.shape)

    for feature in ['actors', 'directors', 'writers', 'genres2', 'keyws', 'production_house']:
        df[feature] = df[feature].apply(lambda x: literal_eval(x))

    for feature in ['actors', 'directors', 'writers', 'genres2', 'keyws', 'production_house']:
        df[feature] = df[feature].apply(lambda x: create_form(x))
    

    movie_id = df[df['title']==movie_name]['imdb_id'].values[0]
    url = "https://www.imdb.com/title/"+movie_id+"/reviews?sort=userRating&dir=desc&ratingFilter=0"
    
    next_page_check=1
    #count=0

    temp_dict={}
    media_details=[]

    user_data_list =[]
    n=0
    
    df_temp=df.copy()


    response=requests.get(url)
    print(url,'\n')
    results = BeautifulSoup(response.content,'lxml')
    reviews_list = results.find('div',{"class":"lister-list"})

    next_page_check=1

    while next_page_check:

        for item in reviews_list.find_all('div',{"class":"lister-item mode-detail imdb-user-review collapsable"}):
            if item=='':
                continue

            try:
                review=item.find('div',{"class":"text show-more__control"}).get_text()
            except:
                continue


            temp_dict={}

            #temp_dict['Title']=k
            temp_dict['Review']=review
            n=n+1

            media_details.append(temp_dict)


            if n>500:
                next_page_check=0
                break

        if 'data-key' in str(results.find('div',{"class":"load-more-data"})) and n<500:
            paginationkey=results.find('div',{"class":"load-more-data"}).get('data-key')

            #print(n)
            #print("Going", n/20, "times")

            headers = {
                'authority': 'www.imdb.com',
                'accept': '*/*',
                'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'cookie': 'session-id=137-0966528-1020544; adblk=adblk_no; uu=eyJpZCI6InV1MDM4MjA5NzNmMjVlNDM4MWEzMjAiLCJwcmVmZXJlbmNlcyI6eyJmaW5kX2luY2x1ZGVfYWR1bHQiOmZhbHNlfSwidWMiOiJ1cjU2MTMyMTg1In0=; beta-control=""; ubid-main=133-7584993-1023227; x-main="oZgQMhoE@Tf3mN28o?i91JLPbldFV4AUxbxncOfFme@@1m0yonWRmp9lv6k9XtzL"; at-main=Atza|IwEBIKP2iVRHaDNkr0jFA7EX-ezZ3j3RXROetfN0pKLnq_uM4Z0jsLkdFY_GExPywdFI9-IbOKAOJSNkuKBpmAXG48C_x3-9PXfKNaqXeVluFceMlTvfubPUxNKfq8DbvxCTN5IFpX1qlWbB8DFHhPhYCGQsd65KaE88kI5XIXCJWz4gZTa-jh8LANvY6V__38PD9pEsJk1mfLn8rZPwyhVZig0K1uR65ZKN50Lf3YdLqod3hobq662iuPhfSCGt6EdrZrU; sess-at-main="ccCCIHSwyB0jTSdvkgrkaV7WVjimD+pWdVKfnHR9aas="; session-id-time=2082787201l; session-token=DzbDj1mvMhnu/hbT9CTZ/WQIEOtRRpXQJNrBn8JGHZVKvH1EBlbjD3dq/+3im/ZNxCuMQaBTxltYiiBMoYAAQBKa3TUCYWPXxLRRvKvrrVLLaw0DLZ/ega839Ew7F0SgMgqlAAAKXqWWLboj/B0Ph1zPi5M4tiZU8zpdtf7sQGsSS6BE9HpaebdsXcyVKSOIReVgClOpWYT5Ynrs4gDTvbVwYyfw6zmHUlhoeg5HTkJlkRwIJkkdfA==; csm-hit=tb:C0WJXFXWMJ2710G6CS4Q+b-BCEXZW8AGRA0WDPWW10K|1679123307651&t:1679123307651&adb:adblk_no',
                'referer': "https://www.imdb.com/title/"+movie_id+"/reviews?sort=userRating&dir=desc&ratingFilter=0",
                'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
                'x-requested-with': 'XMLHttpRequest',
            }

            params = (
                ('sort', 'userRating'),
                ('dir', 'desc'),
                ('ratingFilter', '0'),
                ('ref_', 'undefined'),
                ('paginationKey', paginationkey),
            )

            response=requests.get("https://www.imdb.com/title/"+movie_id+"/reviews/_ajax", headers=headers, params=params)
            results = BeautifulSoup(response.content,'lxml')

            reviews_list = results.find('div',{"class":"lister-list"})

            next_page_check=1

        else:
            next_page_check=0

        
    print(n)
    
    df_test=pd.DataFrame(media_details)
    
    
    df_test['spacy_ent'] = df_test['Review'].apply(lambda x: spacy_ent(x))

    #df_test['ModifiedReview'] = ''
    
    df_test['ModifiedReview'] = df_test.apply(lambda x: spacy_change(x['Review'], x['spacy_ent']), axis=1)
    
    #df_test['porter_tokenized'] = df_test['Review'].apply(lambda x: tokenizer_porter(x))
    
    #df_test['snowball_tokenized'] = df_test['Review'].apply(lambda x: tokenizer_snowball(x))
    
    df_test['tokenized'] = df_test['Review'].apply(lambda x: tokenization(x))
    
    df_test['lemmatized'] = df_test['tokenized'].apply(lambda x: lemmatizer(x))
    
    df_test['stopwords_removed'] = df_test['lemmatized'].apply(lambda x: remove_stop(x))
    
    all_content = ""
    
    df_test['nouns_only'] = df_test['stopwords_removed'].apply(lambda x: nouns_isolated(x))
    
    for review in df_test['nouns_only'].to_list():
        for word in review:
            if word not in ['movie', 'film', "n't", 'watching', 'wa', 's', 'ha', 'watch', 'one', 'doe', movie_name]:
                all_content = all_content + ' ' + word
        else:
            continue
    
    word_cloud = wc.generate(all_content)

    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()
    
    
    
    top_words = sorted(wc.process_text(all_content).items(), key=operator.itemgetter(1), reverse = True)[:50]
    
    print(top_words)

    categories = {'directors':['director','direct','scene','mood','visuals','cinematography','style'],'writers':['story','script','plot','writer','write','character'],'actors':['performance','act','actor','actress','cast','ensemble'],'genres2':[
        'animation',
 'comedy',
 'family',
 'adventure',
 'fantasy',
 'action',
 'crime',
 'drama',
 'thriller',
 'romance',
 'horror',
 'sciencefiction',
 'mystery',
 'music',
 'history',
 'war',
 'western',
 'documentary',
 'tvmovie']}

    weight = {'directors':0, 'writers':0, 'actors':0, 'genres2':0, 'keywords':0}
    more_keyws = []
    
    for tup in top_words:
        if tup[0] in [item for sublist in categories.values() for item in sublist]:
            for cat,list_val in categories.items():
                for val in list_val:
                    if val==tup[0]:
                        weight[cat]+=tup[1]
        else:
            more_keyws.append(tup[0])
            weight['keywords']+=tup[1]
        
    
    print('Category counts:')
    print(weight)

    weight_key = []
    for tup in top_words:
        if tup[0] in [item for sublist in categories.values() for item in sublist]:
            continue
        else:
            weight_key.append(tup[0])

    keyword_weight = 0.4
    production_weight = 0.00

    weight['keywords'] = keyword_weight

    temp_sum = (weight['directors']+weight['writers']+weight['actors']+weight['genres2'])/(1-(weight['keywords']+production_weight))
    
    
    for key, val in weight.items():
        weight[key]=val/temp_sum
        
    print('Weightages created')
    print(weight)
    
    movie_reviews = df_test['ModifiedReview'].tolist()

    all_reviews = ". ".join(movie_reviews)

    all_reviews_text = Text(word_tokenize(all_reviews))
    
    movie_items = df_temp[df_temp['title']==movie_name]['soup'].str.split(' ').values#tolist()
    
    movie_items = movie_items_extend_keyws(movie_items, more_keyws)

    temp_list = []
    score = {}
    num_mentions = {}

    for attr in movie_items:
        temp_list = all_reviews_text.concordance_list(attr,width=1000, lines=1000)
        all_samples = ''
        m=0
        for item in temp_list:
            all_samples = ' '+item.line
            m=m+1

        num_mentions[attr] = m
        score[attr] = get_sentiment_pos(all_samples)

    print(num_mentions)
    final_score = {}

    for key in score:
        final_score[key] = score[key]*num_mentions[key]
    
    print('Finding min')

    min_val = None
    result = None
    for k, v in final_score.items():
        if v and (min_val is None or v < min_val):
            min_val = v
            result = k
    
    print("min =",min_val)

    num_times = {}

    for key in final_score:
        num_times[key] = math.floor(final_score[key]/min_val)
        
    df_temp['keyws'] = df_temp.apply(lambda x: keywords_extend_keyws(x['keyws'], x['title'], movie_name, more_keyws), axis=1)


    for feature in ['actors', 'directors', 'writers', 'genres2', 'keyws', 'production_house']:
        df_temp[feature] = df_temp.apply(lambda x: create_again(x[feature], x['title'],num_times, movie_name), axis=1)
        
    
    print('Figured out num_times:')
    print(num_times)
        
    
    count = CountVectorizer(stop_words='english')
    
    count_directors_matrix = count.fit_transform(df['directors'])

    cosine_sim_directors = cosine_similarity(count_directors_matrix, count_directors_matrix)

    count_writers_matrix = count.fit_transform(df['writers'])

    cosine_sim_writers = cosine_similarity(count_writers_matrix, count_writers_matrix)

    count_actors_matrix = count.fit_transform(df['actors'])

    cosine_sim_actors = cosine_similarity(count_actors_matrix, count_actors_matrix)

    count_genres_matrix = count.fit_transform(df['genres2'])

    cosine_sim_genres = cosine_similarity(count_genres_matrix, count_genres_matrix)

    count_keys_matrix = count.fit_transform(df['keyws'])

    cosine_sim_keys = cosine_similarity(count_keys_matrix, count_keys_matrix)

    count_prod_matrix = count.fit_transform(df['production_house'])

    cosine_sim_prod = cosine_similarity(count_prod_matrix, count_prod_matrix)

    weightage = []

    for k,v in weight.items():
        if k!='keywords':
            weightage.append(v)

    weightage.append(keyword_weight)
    weightage.append(production_weight)
    
    print(weightage)

    cosine_metadata = (cosine_sim_directors*weightage[0]) + (cosine_sim_writers*weightage[1]) + (cosine_sim_actors*weightage[2]) + (cosine_sim_genres*weightage[3]) + (cosine_sim_keys*weightage[4]) + (cosine_sim_prod*weightage[5])

    indices = pd.Series(df_temp.index, index=df_temp['title'])
    
    idx = indices[movie_name]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_metadata[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #movie_indices = [i[0] for i in sim_scores]
    #temp = pd.DataFrame(df_temp[['title','release_date','vote_average']].iloc[movie_indices])


    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:81]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    similarity = [i[1] for i in sim_scores]
    
    #similarity = [i[1] for i in sim_scores[1:n+1]]
    
    answer = pd.DataFrame(df_temp[['title','release_date','vote_average']].iloc[movie_indices])
    
    answer['sim'] = similarity
    
    movie_score = df_temp['vote_average'][df_temp['title']==movie_name].values[0]
    
    print('movie score = ', movie_score)
    
    threshold = movie_score-((1-(10-movie_score)/10)*(2))
    print("Threshold=", threshold)
    
    #df_removed = answer[answer['vote_average']<threshold]
    
    df_retained = answer[answer['vote_average']>threshold]
    
    #print("Removed movies=", df_removed.head(20))
    print("Retained movies=", df_retained.head(20))
    
    
    #html_table = build_table(answer[['title','release_date']].head(10), 'red_dark', even_bg_color='light_red')
    
    #with open('styled_table.html', 'w') as f:
    #    f.write(html_table)

    #Return the top 10 most similar movies
    #pd.DataFrame(df_temp[['title','release_date']].iloc[movie_indices])
    
    result = df_retained[['title','release_date']].head(10)
    result.index = np.arange(1, len(result) + 1)
    dfStyler = result.style.set_properties(**{'text-align': 'left', 'background-color': '#fec5b2', 'font-size': '20pt'})
    dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left'),('font-size', '30pt')])])
    
    return dfStyler    


# ## Test

# In[56]:


#recommendation_engine('2012')


# In[ ]:




