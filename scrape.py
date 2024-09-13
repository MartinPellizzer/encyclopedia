import os
import time
import random
import datetime
import shutil
from urllib.request import urlretrieve

from Bio import Entrez
import metapub

from oliark import file_write
from oliark import json_write
from oliark import csv_read_rows_to_json


proj = 'terrawhisper'

vault = '/home/ubuntu/vault'
pubmed_folderpath = f'{vault}/{proj}/studies/pubmed'

Entrez.email = 'martinpellizzer@gmail.com'
# sort_by = 'pub_date'
sort_by = 'relevance'
datetypes = ['mdat', 'pdat', 'edat']
datetype = datetypes[2]
years = [year for year in range(2024, 1785, -1)]
yesterday = datetime.datetime.now() - datetime.timedelta(1)
yesterday = datetime.datetime.strftime(yesterday, '%Y/%m/%d')

actions_num_total = 0

def create_folder(folderpath):
    chunk_curr = ''
    for chunk in folderpath.split('/'):
        chunk_curr += f'{chunk}/'
        try: os.makedirs(chunk_curr)
        except: continue

def get_ids(query, year=None, date=None, retmax=50):
    if date:
        handle = Entrez.esearch(db='pubmed', term=query, retmax=retmax, sort=sort_by, datetype=datetype, mindate=date, maxdate=date)
    elif year:
        handle = Entrez.esearch(db='pubmed', term=query, retmax=retmax, sort=sort_by, datetype=datetype, mindate=year, maxdate=year)
    else:
        handle = Entrez.esearch(db='pubmed', term=query, retmax=retmax, sort=sort_by)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

def fetch_details(pmid):
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml')
    try: records = Entrez.read(handle)
    except: 
        handle.close()
        return None
    handle.close()
    return records

def scrape_pubmed_jsons(query, year=None, date=None):
    global actions_num_total
    ailment_slug = ailment_name.replace(' ', '-')
    pmid_list = get_ids(query, year=year, retmax=9999)
    query_folderpath = f'{pubmed_folderpath}/{ailment_slug}'
    if pmid_list:
        create_folder(f'{query_folderpath}/json')
        create_folder(f'{query_folderpath}/pdf')
        for pmid_i, pmid in enumerate(pmid_list):
            print(f'{pmid_i}/{len(pmid_list)} - {query} - {year} - {date}')
            
            json_done_pmids = []
            for filename in os.listdir(f'{query_folderpath}/json'):
                filename_raw = filename.split('.')[0]
                json_done_pmids.append(filename_raw)
            if pmid not in json_done_pmids:
                try: details = fetch_details(pmid)
                except: continue
                json_write(f'{query_folderpath}/json/{pmid}.json', details)
                time.sleep(random.randint(1, 3))
                actions_num_total += 1
            
            pdf_done_pmids = []
            for filename in os.listdir(f'{query_folderpath}/pdf'):
                filename_raw = filename.split('.')[0]
                pdf_done_pmids.append(filename_raw)
            if pmid not in pdf_done_pmids:
                try: src = metapub.FindIt(pmid)
                except:
                    file_write(f'{query_folderpath}/pdf/{pmid}', '')
                    continue
                print(src.pma.title)
                if src.url:
                    print(src.url)
                    try: urlretrieve(src.url, f'{query_folderpath}/pdf/{pmid}.pdf')
                    except: file_write(f'{query_folderpath}/pdf/{pmid}.pdf', '')
                else:
                    file_write(f'{query_folderpath}/pdf/{pmid}.pdf', '')
                    print(src.reason)
                print()
                time.sleep(random.randint(1, 3))
                actions_num_total += 1
                
            if actions_num_total >= 1000:
                actions_num_total = 0
                time.sleep(random.randint(450, 750))
    else:
        print('no article found')
    time.sleep(random.randint(1, 3))

ailments = csv_read_rows_to_json(f'/home/ubuntu/vault/terrawhisper/csvs/ailments.csv')
for ailment in ailments:
    ailment_name = ailment['ailment_name']
    organ_name = ailment['organ_name']
    print(ailment_name, organ_name)

    for year in years:
        scrape_pubmed_jsons(ailment_name, year)
    
