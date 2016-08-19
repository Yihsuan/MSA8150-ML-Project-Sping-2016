'''
Created on Mar 25, 2016

@author: Yi-Hsuan Su
'''
from __future__ import division
from pyechonest import config
config.ECHO_NEST_API_KEY="GD5FDSYCME8UKHKJY"

from pyechonest import song, artist, track
from pyechonest.util import EchoNestAPIError, EchoNestIOError
import urllib2, pprint, json, time
from socket import timeout



def create_artist_list():
    artist_list = {}
    searched = {}
    
    while len(artist_list) <= 10000:
        try:
            if len(artist_list) == 0:
                for i in artist.top_hottt(results = 100):
                    artist_list[i.name] = 1
            
            elif len(searched) == len(artist_list):
                print 'End of expanding the artist list'
            
            else:
                for i in [a for a in artist_list.keys() if a not in searched]:
                    searched[i] = 1
                    a = artist.Artist(i)
                    for j in a.get_similar(results = 100):
                        if j not in artist_list:
                            artist_list[j.name] = 2
        except EchoNestAPIError:
            print '# of searched artists:', len(searched)
            print '# of artists', len(artist_list)
            print '------'
            #print artist_list.keys()
            time.sleep(60)
            
    print 'Final # of artists', len(artist_list)
    #pprint.pprint(artist_list)
    
    with open('artist_list.txt', mode = 'w') as f:
        f.write('\n'.join(artist_list.keys()).encode('utf8'))
        
def create_song_list():
    with open('artist_list(10k).txt', 'r') as f:
        artist_list = f.read().split('\n')
    
    #artist_list = artist_list[:100]
    print 'Now run on', len(artist_list), 'artists...'
    
    artist_done = {}
    song_list = []
    result_num = 1
    start_index = 0
    
    while len(artist_done) != len(artist_list):
        try:
            for i in [aa for aa in artist_list if aa not in artist_done]:
                print 'now collecting songs of', i
                a = artist.Artist(i) #0-1
                
                search_song = True 
                while search_song:
                    ls = a.get_songs(results = result_num, start=start_index) #1
                    #print 'start_index:', start_index
                    #print '# of song found', len(ls)
                    if len(ls)>=1:
                        song_song = ls[0]
                        title = song_song.title 
                        idd = song_song.id;
                        s_hottness = song_song.song_hotttnesss; # 2
                        a_hottness = song_song.artist_hotttnesss; # 3
                        a_familarity = song_song.artist_familiarity; #4
                        summary = song_song.audio_summary ; # 5
                        #analysis = urllib2.urlopen(summary['analysis_url'])
                        #analysis = eval(analysis.read())
                            
                        song_list.append({'artist':i, 'title': title,
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
                        print idd
                    
                    if len(ls) < result_num:
                        start_index = 0
                        search_song = False
                    else: start_index += result_num
                    
                artist_done[i] = 1
               
        except EchoNestAPIError:
            print '---sleeping for 60 sec---'
            print 'Status::'
            print '# of searched artist:', len(artist_done)
            print '# of songs:', len(song_list)
            print '\n'
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)
                       
            time.sleep(60)
        except EchoNestIOError as e:
            print '***caught a IOError***'
            print e
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)
        except timeout:
            print '***caught a timeout***'
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)
    
    print 'total # of songs in file', len(song_list)

if __name__ == '__main__':
    #create_artist_list()
    t = time.time()
    
    create_song_list()
    
    spent = time.time()-t
    hour = int(spent//3600)
    remain = int(spent%3600)
    minute = int(remain//60)
    remain = int(remain%60)
    
    print 'Elapsed time:',hour, ':', minute, ':', remain
    
    
    