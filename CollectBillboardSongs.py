'''
Created on Apr 7, 2016

@author: Yi-Hsuan Su
'''
from __future__ import division
from pyechonest import config
config.ECHO_NEST_API_KEY="BS5QS7D1CSSGO7QP6"

from pyechonest import song, artist, track
from pyechonest.util import EchoNestAPIError, EchoNestIOError
import urllib2, pprint, json, time
from socket import timeout
import pandas as pd

def get_billboard_song_data():

    df = pd.read_csv('billboard_songs_top10.csv')
    df = df[['Title', 'Artist']]
    df = df.drop_duplicates()
    #print 'Now run on', len(artist_list), 'artists...'
    
    song_done = []
    song_list = []
    #result_num = 1
    #start_index = 0
    
    i = 0
    while len(song_done) != len(df):
        try:    
            t = df.iloc[i,0]
            a = df.iloc[i,1]
            print 'Now collecting...' + t + ' by ' + a
            ls = song.search(title = t, artist = a)
            if len(ls)>=1:
                song_song = ls[0]
                # title = song_song.title 
                idd = song_song.id;
                s_hottness = song_song.song_hotttnesss; # 2
                a_hottness = song_song.artist_hotttnesss; # 3
                a_familarity = song_song.artist_familiarity; #4
                summary = song_song.audio_summary ; # 5                        
                song_list.append({'artist':a, 'title': t,
                                'song_id': idd,
                                'song_hottness': s_hottness,
                                'artist_hottness': a_hottness,
                                'artist_familiarity': a_familarity,
                                'danceability': summary['danceability'],
                                'duration': summary['duration'],
                                'energy': summary['energy'],
                                'key': summary['key'],
                                'liveness': summary['liveness'],
                                'loudness': summary['loudness'],
                                'speechiness': summary['speechiness'],
                                'acousticness': summary['acousticness'],
                                'instrumentalness': summary['instrumentalness'],
                                'mode': summary['mode'],
                                'time_signature': summary['time_signature'],
                                'tempo': summary['tempo']
                            }) 
                print '>>> song found'                
            song_done.append({'artist': a, 'title': t})
            i += 1
                
        except EchoNestAPIError:
            print '---sleeping for 60 sec---'
            print 'Status::'
            print '# of songs queried:', len(song_done)
            print round(len(song_done)*100/len(df), 2), '% songs queried'
            print '# of songs found:', len(song_list)
            print round(len(song_list)*100/len(df), 2), '% songs queried'
            print '\n' 
            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)           
            time.sleep(60)
        except EchoNestIOError as e:
            print '***caught a IOError***'
            print e
            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)
        except timeout:
            print '***caught a timeout***'
            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)
    
    
    print 'total # of songs in file', len(song_list)
      
    with open('billboard_songs.json', 'w') as f:
        json.dump(song_list, f)

if __name__ == '__main__':
    #create_artist_list()
    t = time.time()
    
    get_billboard_song_data()
    
    spent = time.time()-t
    hour = int(spent//3600)
    remain = int(spent%3600)
    minute = int(remain//60)
    remain = int(remain%60)

