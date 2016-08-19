'''
Created on Apr 14, 2016

@author: Yi-Hsuan Su
'''
from __future__ import division
from pyechonest import config
config.ECHO_NEST_API_KEY="GD5FDSYCME8UKHKJY"

#from pyechonest import song, artist, track
#from pyechonest.util import EchoNestAPIError, EchoNestIOError
import urllib2, json, time
from urllib2 import HTTPError
from socket import timeout
import pandas as pd
import re, sys

'''
all_genre_url = 'http://developer.echonest.com/api/v4/artist/list_genres?api_key=GD5FDSYCME8UKHKJY&format=json'
all_genre_content = eval(urllib2.urlopen(all_genre_url).read())['response']['genres']
print "# of genres: ", len(all_genre_content)
for genre in all_genre_content:
    print genre['name']
'''
'''
collected = eval(open('artists_genres.json').read())
collected_artists = collected.keys()
'''
df1 = pd.read_json('songs.json', orient= 'records')[['artist']]
df2 = pd.read_csv('us_billboard_1941_2015.csv')[['Artist']]

df2.columns = ['artist'] 

artists = pd.concat([df1[['artist']], df2[['artist']]])
artists = artists.drop_duplicates()

#artists = [x for x in artists['artist'] if x not in collected_artists]
print '# of artists:', len(artists)

all_artists_genres = {}
artists_failed = []
for i in artists['artist']:
    try:
        artist = i
        i = re.sub(r'[\W]', '%20', i)
        url = 'http://developer.echonest.com/api/v4/artist/profile?api_key=GD5FDSYCME8UKHKJY&name='+i.strip().replace(' ', '%20')+'&bucket=genre&format=json'
        artist_genre = eval(urllib2.urlopen(url).read())
        artist_genre = artist_genre['response']['artist']['genres']
        artist_genre_list = []
        for g in artist_genre:
            artist_genre_list.append(g['name'])
        
        all_artists_genres[artist] = artist_genre_list
    except HTTPError as e:
        print '***', e.msg, '***'
        if e.msg == 'Too Many Requests':
            print len(all_artists_genres), 'artists' ,str(round(len(all_artists_genres)*100/len(artists), 2))+'% completed'
        else:
            print 'Can\'t get the genres of artist:', artist
            artists_failed.append(artist)
        print '--------------'
        time.sleep(60)
    except KeyError as k:
        print 'key error'
        print '--------------'
        pass
    except:
        print 'Can\'t get the genres of artist:', artist
        artists_failed.append(artist)
        print '--------------'
        
    
        with open('artists_genres.json', 'w') as f:
            json.dump(all_artists_genres, f)
            
        with open('artists_genres.json', 'w') as f:
            f.write('\n'.join(list(map(str, artists_failed))))
            
with open('artists_genres.json', 'w') as f:
    json.dump(all_artists_genres, f)
with open('artists_genres.json', 'w') as f:
    f.write('\n'.join(list(map(str, artists_failed))))
print len(all_artists_genres), 'artists' ,str(round(len(all_artists_genres)*100/len(artists), 2))+'% completed'
print '--------------'
print "Done"
        
        
        