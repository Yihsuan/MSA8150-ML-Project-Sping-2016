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


class Status:
    def __init__(self, collect_func):
        self.data = {}
        self.log_name = None
        self.mode = 'a'
        self.func = collect_func

    def status_create_artist_list(self):
        if self.data and self.log_name:
            with open(self.log_name, self.mode) as f:
                f.write('\nStatus:')
                f.write('\n# of artist in the list: ' + str(len(self.data)))

    def log_status(self):
        if self.func == 'create_artist_list':
            self.status_create_artist_list()


class Log(Status):
    def __init__(self, collect_func, log_name=None):
        super(Log, self).__init__(collect_func)

        if not os.path.isdir('log'):
            os.mkdir('log')
        self.log_name = log_name
        self.mode = 'a'
        self.beg_t = time.time()

    def log_heading(self):
        if not self.log_name:
            return

        self.mode = 'w+'
        with open(self.log_name, mode=self.mode) as f:
            f.write('=' * 20 + 'Date: ' + str(date.today()) + '=' * 20)
            f.close()

    def log_elapsed_time(self):
        if not self.log_name:
            return

        h, m, s = calculate_elapsed_time(self.beg_t)

        with open(self.log_name, self.mode) as f:
            f.write('\nElapsed time: ' + str(h) + ':' + str(m) + ':' + str(s))

    def log_ending(self):
        if not self.log_name:
            return

        with open(self.log_name, self.mode) as f:
            f.write('=' * 20 + 'END' + '=' * 20)

        self.elapsed_time()


class SongsCollection(Log):
    def __init__(self):
        self.data = None
        self.n = 10000
        self.list_names = {'create_artist_list': 'artist_list.txt',
                           'create_song_list': 'song_list.txt',
                           'get_billboard_song_data': 'billboard_song_list.txt'}

        self.log_names = {'create_artist_list': 'create_artist_list.log',
                          'create_song_list': 'create_song_list.log',
                          'get_billboard_song_data': 'echonest_billboard_songs.log'}

    def create_artist_list(self, n=None, list_name=None, log_name=None):
        """
        n is desired number of artists in the list
        """

        if not n:
            n = self.n
        if not list_name:
            list_name = self.list_names['create_artist_list']
        if not log_name:
            log_name = self.log_names['create_artist_list']

        begin_t = time.time()

        super(SongsCollection, self).__init__(log_name)
        self.log_heading()

        artist_list = {}
        searched = {}

        # get the 100 hottest artist
        artist_list.update({i.name: 0 for i in artist_list.top_hottt(result=100)})

        while len(artist_list) <= n:
            try:
                if len(searched) == len(artist_list):
                    with open(log_name, 'a') as f:
                        f.write('\nEnd of expanding the artist list')
                else:
                    for i in [a for a in artist_list.keys() if a not in searched]:
                        a = artist.Artist(i)
                        for j in a.get_similar(results=100):
                            if j not in artist_list:
                                artist_list[j.name] = 0
                        searched[i] = 1

            except EchoNestAPIError:
                self.log_elapsed_time()

                self.data = artist_list

                with open(list_name, mode='w') as f:
                    f.write('\n'.join(self.data.keys()).encode('utf8'))

                self.log_status()

                time.sleep(60)

        self.data = artist_list

        with open(list_name, mode='w') as f:
            f.write('\n'.join(self.data.keys()).encode('utf8'))

        self.log_ending()
        self.log_status()

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
