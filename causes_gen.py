import os
import sys
import json

from oliark import csv_read_rows_to_json, csv_read_rows
from oliark import json_read, json_write

from oliark_llm import llm_reply

vault = '/home/ubuntu/vault'

model = f'{vault}/llms/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
encyclopedia_folderpath = f'{vault}/terrawhisper/encyclopedia'

ailments_folderpath = f'{encyclopedia_folderpath}/jsons'

def causes(ailment_filepath):
    ailment_slug = ailment_filepath.split('/')[-1].split('.')[0].strip().lower()
    ailment_name = ailment_slug.replace('-', ' ')
    
    data = json_read(ailment_filepath)
    if 'causes' not in data:
    # if True:
        prompt = f'''
            Write a numbered list of the most common causes of the following ailment: {ailment_name}. 
            Order the list by the most frequent cause.
            Reply only with the names of the causes, don't add descriptions.
            Reply in as few words as possible.
            Include only one cause per list item.
            Don't use brackets.
        '''
        causes = []
        for i in range(100):
            reply = llm_reply(prompt, model, max_tokens=256)
            lines = []
            for line in reply.split('\n'):
                line = line.strip()
                if line == '': continue
                if not line[0].isdigit(): continue
                if '. ' not in line: continue
                line = '. '.join(line.split('. ')[1:])
                line = line.strip()
                if line == '': continue
                lines.append(line)
            for line in lines:
                found = False
                for cause in causes:
                    if line in cause['name']: 
                        cause['mentions'] += 1
                        found = True
                        break
                if not found:
                    causes.append({
                        'name': line, 
                        'mentions': 1, 
                    })
        causes = sorted(causes, key=lambda x: x['mentions'], reverse=True)
        print('***********************')
        print('***********************')
        print('***********************')
        for cause in causes:
            print(cause)
        print('***********************')
        print('***********************')
        print('***********************')
        data['causes'] = causes
        json_write(ailment_filepath, data)


def symptoms(ailment_filepath):
    ailment_slug = ailment_filepath.split('/')[-1].split('.')[0].strip().lower()
    ailment_name = ailment_slug.replace('-', ' ')
    
    data = json_read(ailment_filepath)
    key = 'symptoms'
    if key not in data:
    # if True:
        prompt = f'''
            Write a numbered list of the most common {key} of the following ailment: {ailment_name}. 
            Order the list by the most frequent cause.
            Reply only with the names of the causes, don't add descriptions.
            Reply in as few words as possible.
            Include only one cause per list item.
            Don't use brackets.
        '''
        causes = []
        for i in range(100):
            reply = llm_reply(prompt, model, max_tokens=256)
            lines = []
            for line in reply.split('\n'):
                line = line.strip()
                if line == '': continue
                if not line[0].isdigit(): continue
                if '. ' not in line: continue
                line = '. '.join(line.split('. ')[1:])
                line = line.strip()
                if line == '': continue
                lines.append(line)
            for line in lines:
                found = False
                for cause in causes:
                    if line in cause['name']: 
                        cause['mentions'] += 1
                        found = True
                        break
                if not found:
                    causes.append({
                        'name': line, 
                        'mentions': 1, 
                    })
        causes = sorted(causes, key=lambda x: x['mentions'], reverse=True)
        print('***********************')
        print('***********************')
        print('***********************')
        for cause in causes:
            print(cause)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = causes
        json_write(ailment_filepath, data)


for ailment_filename in os.listdir(ailments_folderpath):
    ailment_filepath = f'{ailments_folderpath}/{ailment_filename}'
    causes(ailment_filepath)
    symptoms(ailment_filepath)
