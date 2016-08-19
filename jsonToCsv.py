'''
Created on Apr 1, 2016

@author: Yi-Hsuan Su
'''

import csv, json, codecs

'''change the path of input and output files before start converting the json file'''

with codecs.open(r'songs.json', encoding='utf-8') as f:
    jsonf = json.loads(f.read())
    
print type(jsonf)

f = csv.writer(open('songs_0412.csv', 'wb+'))
# use encode to convert non-ASCII characters

field = jsonf[0].keys()

f.writerow(field)


for item in jsonf:
    item['artist'] = item['artist'].encode('utf-8')
    item['title'] = item['title'].encode('utf-8')
    values = [item[x] for x in item]
    f.writerow(values)        
    
