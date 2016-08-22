"""
Created on Apr 7, 2016

@author: Yi-Hsuan Su
"""
from __future__ import division
from pyechonest import config
from pyechonest import song, artist, track
from pyechonest.util import EchoNestAPIError, EchoNestIOError
import json, time
from socket import timeout
import pandas as pd

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


def get_billboard_song_data():

    fname = 'billboard_songs_top10.csv'
    begin_t = time.time()

    df = pd.read_csv(fname)
    df = df[['Title', 'Artist']]
    df = df.drop_duplicates()
    with open('echonest_billboard_songs.log', 'w+') as f:
        f.write('Read from ' + fname + 'that has ' + str(len(df)) + 'songs')

    song_done = []
    song_list = []

    i = 0
    while len(song_done) != len(df):
        try:    
            t = df.iloc[i, 0]   # Title
            a = df.iloc[i, 1]   # Artist
            ls = song.search(title=t, artist=a)
            with open('echonest_billboard_songs.log', 'a') as f:
                f.write('\n- ' + t + ' by ' + a)

            if len(ls) >= 1:
                song_song = ls[0]
                # title = song_song.title 
                idd = song_song.id
                s_hottness = song_song.song_hotttnesss
                a_hottness = song_song.artist_hotttnesss
                a_familarity = song_song.artist_familiarity
                summary = song_song.audio_summary
                song_list.append({'artist': a,
                                  'title': t,
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
                with open('echonest_billboard_songs.log', 'a') as f:
                    f.write('>>> song found')
            else:
                with open('echonest_billboard_songs.log', 'a') as f:
                    f.write('>>> can\'t find the song')
            song_done.append({'artist': a, 'title': t})
            i += 1
                
        except EchoNestAPIError:
            hrs, mins, secs = calculate_elapsed_time(begin_t)

            with open('echonest_billboard_songs.log', 'a') as f:
                f.write('\nStatus:')
                f.write('\nElapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
                f.write('\n' + str(round(len(song_done)*100/len(df), 2)) + ' % songs queried')
                f.write('\n' + str(round(len(song_list)*100/len(df), 2)) + ' % songs found')
                f.write('\n------')

            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)           
            time.sleep(60)

        except EchoNestIOError as e:
            with open('echonest_billboard_songs.log', 'a') as f:
                f.write('\n***caught a IOError***\n' + str(e))
            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)
        except timeout:
            with open('echonest_billboard_songs.log', 'a') as f:
                f.write('\n***caught a timeout***')

            with open('billboard_songs.json', 'w') as f:
                json.dump(song_list, f)

    hrs, mins, secs = calculate_elapsed_time(begin_t)

    with open('echonest_billboard_songs.log', 'a') as f:
        f.write('\nTotal elapsed time: ' + str(hrs) + ':' + str(mins) + ':' + str(secs))
        f.write('\nTotal # of songs in file: ' + str(len(song_list)))

    with open('billboard_songs.json', 'w') as f:
        json.dump(song_list, f)

if __name__ == '__main__':

    get_billboard_song_data()
