'''
Created on Mar 25, 2016

@author: Yi-Hsuan Su
'''
from __future__ import division
from pyechonest import config
from pyechonest import song, artist, track
from pyechonest.util import EchoNestAPIError, EchoNestIOError
import json, time
from socket import timeout
from datetime import date

"""
TODO: Use Echo Nest API Key
"""
config.ECHO_NEST_API_KEY = None


def calculate_elapsed_time(begin):
    spent = time.time() - begin
    hour = int(spent // 3600)
    remain = int(spent % 3600)
    minute = int(remain // 60)
    second = int(remain % 60)

    return hour, minute, second


def create_artist_list(n=10000):
    """
    n is desired number of artists in the list
    """
    begin_t = time.time()

    artist_list = {}
    searched = {}

    with open('create_artist_list.log', 'w+') as f:
        f.write('=' * 20 + 'Date: ' + str(date.today()) + '=' * 20)
        f.close()

    while len(artist_list) <= n:
        try:
            if len(artist_list) == 0:
                # get the 100 hottest artist
                for i in artist.top_hottt(results=100):
                    artist_list[i.name] = 1

            elif len(searched) == len(artist_list):
                with open('create_artist_list.log', 'a') as f:
                    f.write('\nEnd of expanding the artist list')
            else:
                for i in [a for a in artist_list.keys() if a not in searched]:
                    searched[i] = 1
                    a = artist.Artist(i)
                    for j in a.get_similar(results=100):
                        if j not in artist_list:
                            artist_list[j.name] = 2

        except EchoNestAPIError:

            hrs, mins, secs = calculate_elapsed_time(begin_t)

            with open('create_artist_list.log', 'a') as f:
                f.write('\nElapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
                f.write('\n# of searched artists: ' + str(len(searched)))
                f.write('\n# of artist in the list: ' + str(len(artist_list)))
                f.write('\n------')
                f.close()
            time.sleep(60)

    hrs, mins, secs = calculate_elapsed_time(begin_t)

    with open('create_artist_list.log', 'a') as f:
        f.write('\nTotal elapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
        f.write('\nFinal # of artists: ' + str(len(artist_list)))

    with open('artist_list.txt', mode='w') as f:
        f.write('\n'.join(artist_list.keys()).encode('utf8'))


def create_song_list():
    begin_t = time.time()

    with open('artist_list(10k).txt', 'r') as f:
        artist_list = f.read().split('\n')
    
    with open('create_song_list.log', 'w+'):
        f.write('=' * 20 + 'Date: ' + str(date.today()) + '=' * 20)
        f.write('\nNow running on' + str(len(artist_list)) + 'artists...')
        f.close()

    artist_done = {}
    song_list = []
    result_num = 1
    start_index = 0

    while len(artist_done) != len(artist_list):
        try:
            for i in [aa for aa in artist_list if aa not in artist_done]:
                with open('create_song_list.log', 'a') as f:
                    f.write('\nNow collecting songs of' + i)

                a = artist.Artist(i)

                search_song = True

                while search_song:
                    ls = a.get_songs(results=result_num, start=start_index)
                    if len(ls) >= 1:
                        song_song = ls[0]
                        title = song_song.title 
                        idd = song_song.id
                        s_hottness = song_song.song_hotttnesss
                        a_hottness = song_song.artist_hotttnesss
                        a_familiarity = song_song.artist_familiarity
                        summary = song_song.audio_summary
                        #analysis = urllib2.urlopen(summary['analysis_url'])
                        #analysis = eval(analysis.read())
                            
                        song_list.append({'artist': i,
                                          'title': title,
                                          'song_id': idd,
                                          'song_hottness': s_hottness,
                                          'artist_hottness': a_hottness,
                                          'artist_familiarity': a_familiarity,
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

                    if len(ls) < result_num:
                        start_index = 0
                        search_song = False
                    else:
                        start_index += result_num
                    
                artist_done[i] = 1
               
        except EchoNestAPIError:
            hrs, mins, secs = calculate_elapsed_time(begin_t)

            with open('create_song_list.log', 'a') as f:
                f.write('\nStatus:')
                f.write('\nElapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
                f.write('\n# of searched artist: ' + str(len(artist_done)))
                f.write('\n# of songs: ' + str(len(song_list)))
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)
                       
            time.sleep(60)

        except EchoNestIOError as e:
            with open('create_song_list.log', 'a') as f:
                f.write('\n***caught a IOError***\n' + str(e))
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)

        except timeout:
            with open('create_song_list.log', 'a') as f:
                f.write('\n***caught a timeout***')
            with open('songs.json', 'w') as f:
                json.dump(song_list, f)

    hrs, mins, secs = calculate_elapsed_time(begin_t)

    with open('create_song_list.log', 'a') as f:
        f.write('\nTotal elapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
        f.write('\nTotal # of songs in file' + str(len(song_list)))


if __name__ == '__main__':

    create_artist_list()
    create_song_list()
