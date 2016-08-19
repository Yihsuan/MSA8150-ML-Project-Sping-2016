
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

artistGenre = pd.read_csv('data/artists_genres.csv')


# In[12]:

artistGenre_bin = []


# In[7]:

rock_keywords = ['rock', 'adult standards', 'metal', 'punk', 'heavy', 'wave', 'beach music', 'british invasion', 'mellow', 
                 'beat', 'post-grung', 'grunge', 'madchester', 'surf', 'zolo']
pop_keywords = ['pop', 'christmas', 'new romantic', 'easy listening', 'boy band', 'girl group',
                'beach music', 'lilith', 'british invasion', 'beat', 'wave', 'madchester']
hiphop_keywords = ['hip hop', 'rap', 'trap', 'crunk', 'juggalo', 'jerk', 'freestyle', 'new jack', 'hurban', 'hyphy']
rb_keywords = ['r&b', 'motown', 'soul', 'urban contemporary', 'quiet storm', 'funk', 'doo-wop', 'blues', 'jazz', 'swing',
              'cabaret', 'reggae', 'beach music', 'big band', 'exotica']
country_keywords = ['country', 'nashville', 'western', 'honky tonk', 'lilith']
dance_keywords = ['dance', 'electr', 'disco', 'hi hrg', 'house', 'lounge', 'edm', 'indietronica', 'rave', 'big room', 'big beat']
alternative_keywords = ['alternative', 'indie', 'folk', 'singer-songwriter', 'grunge', 'lilith', 'mellow', 'post-grung']


# In[13]:

for row in artistGenre.iterrows():
    artist = row[1]['artist']
    genres = eval(row[1]['genre'])
    genres_str = ';'.join(genres)
    
    a = {}
    a['artist'] = artist
    a['rock'] = 0
    a['pop'] = 0
    a['hip hop'] = 0
    a['r&b'] = 0
    a['country'] = 0
    a['dance'] = 0
    a['alternative'] = 0
    
    for w in rock_keywords:
        if w == 'beat' and w in genres_str and 'big beat' not in genres_str:
            a['rock'] = 1 
            break
        elif w in genres_str:
            a['rock'] = 1
            break
    for w in pop_keywords:
        if w == 'beat' and w in genres_str and 'big beat' not in genres_str:
            a['pop'] = 1 
            break
        elif w in genres_str:
            a['pop'] = 1
            break
    for w in hiphop_keywords:
        if w in genres_str:
            a['hip hop'] = 1
            break
    for w in rb_keywords:
        if w in genres_str:
            a['r&b'] = 1
            break
    for w in country_keywords:
        if w in genres_str:
            a['country'] = 1
            break
    for w in dance_keywords:
        if w in genres_str:
            a['dance'] = 1
            break
    for w in alternative_keywords:
        if w in genres_str:
            a['alternative'] = 1
            break
        
    artistGenre_bin.append(a)


# In[15]:

artistGenre_bin = pd.DataFrame(artistGenre_bin)


# In[16]:

artistGenre_bin.to_csv('data/artist_genre_bin.csv', index = False)

