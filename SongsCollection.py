from __future__ import division
from pyechonest import config
from pyechonest import song, artist, track
from pyechonest.util import EchoNestAPIError, EchoNestIOError
from socket import timeout
from datetime import date
import pandas as pd
import json
import time
import sys
import os

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


class Log:
    def __init__(self):
        if not os.path.isdir('log'):
            os.mkdir('log')

    def heading(self, log_name, mode='w+'):
        with open(log_name, mode=mode) as f:
            f.write('=' * 20 + 'Date: ' + str(date.today()) + '=' * 20)
            f.close()


class SongsCollection(Log):
    def __init__(self):
        super(SongsCollection, self).__init__()

        self.n = 10000
        self.log_names = {'create_artist_list': 'create_artist_list.log',
                          'create_song_list': 'create_song_list.log',
                          'get_billboard_song_data': 'echonest_billboard_songs.log'}

    def create_artist_list(self, n=None):
        """
        n is desired number of artists in the list
        """
        if not n:
            n = self.n

        begin_t = time.time()

        artist_list = {}
        searched = {}

        log_name = self.log_names[sys._getframe().f_code.co_name]
        self.heading(log_name)

        while len(artist_list) <= n:
            try:
                if len(artist_list) == 0:
                    # get the 100 hottest artist
                    for i in artist.top_hottt(results=100):
                        artist_list[i.name] = 1

                elif len(searched) == len(artist_list):
                    with open(log_name, 'a') as f:
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
                    f.write('\nStatus:')
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



    def create_song_list(self):
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
                            # analysis = urllib2.urlopen(summary['analysis_url'])
                            # analysis = eval(analysis.read())

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
                    f.write('\n------')
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

    def get_billboard_song_data(self):

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
                t = df.iloc[i, 0]  # Title
                a = df.iloc[i, 1]  # Artist
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
                    f.write('\n' + str(round(len(song_done) * 100 / len(df), 2)) + ' % songs queried')
                    f.write('\n' + str(round(len(song_list) * 100 / len(df), 2)) + ' % songs found')
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
