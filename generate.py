import os
import sys
import csv
import json

import chromadb
from chromadb.utils import embedding_functions

from oliark import csv_read_rows_to_json, csv_read_rows
from oliark import json_read, json_write
from oliark_llm import llm_reply

vault = '/home/ubuntu/vault'
llms_path = f'{vault}/llms'

model = f'{llms_path}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
model_validator_filepath = f'{llms_path}/Llama-3-Partonus-Lynx-8B-Intstruct-Q4_K_M.gguf'

proj_name = 'terrawhisper'
db_path = f'{vault}/{proj_name}/database/{proj_name}'
encyclopedia_folderpath = f'{vault}/{proj_name}/encyclopedia'
jsons_100_folderpath = f'{vault}/{proj_name}/encyclopedia/jsons'
jsons_1000_folderpath = f'{vault}/{proj_name}/encyclopedia/jsons-1000'
jsons_folderpath = jsons_100_folderpath
csvs_folderpath = f'{vault}/{proj_name}/csvs'

ailments = csv_read_rows_to_json(f'{csvs_folderpath}/ailments.csv')
ailments_folderpath = f'{encyclopedia_folderpath}/jsons'

## init croma
chroma_client = chromadb.PersistentClient(path=db_path)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-mpnet-base-v2', 
    device='cuda',
)

# plants = csv_read_rows_to_json(f'{vault}/terrawhisper/csvs/trefle.csv')
plants = csv_read_rows_to_json(f'{vault}/wcvp_dwca/wcvp_taxon.csv', delimiter = '|')


def validate_names_scientific(reply):
    names_scientific = []
    for line in reply.split('\n'):
        line = line.strip()
        if line == '': continue
        for plant in plants:
            name_scientific = plant['scientfiicname']
            if name_scientific.lower().strip() in line.lower().strip():
                if len(name_scientific.split(' ')) > 1:
                    print('++++++++++++++++++++++++++++++++++++++++')
                    print(name_scientific)
                    print('++++++++++++++++++++++++++++++++++++++++')
                    names_scientific.append(name_scientific)
                    break
    return names_scientific


def count_mentions():
    for json_filename in os.listdir(jsons_folderpath):
        json_filepath = f'{jsons_folderpath}/{json_filename}'
        data = json_read(json_filepath)
        total = 0
        for obj in data:
            total += obj['plant_mentions']
        print(f'{json_filename}: {total}')


def llm_validate(question, context, answer):
    prompt = f'''
    Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

    --
    QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
    {question}

    --
    DOCUMENT:
    {context}

    --
    ANSWER:
    {answer}

    --

    Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
    {{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
    '''
    reply = llm_reply(prompt, model_validator_filepath, max_tokens=256)
    return reply


def retrieve_docs(query):
    collection_name = 'medicinal-plants'
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef,
    )
    n_results = 100
    results = collection.query(query_texts=[query], n_results=n_results)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    return documents, metadatas


########################################################################
## main
########################################################################
systems_organs_ailments_filepath = f'{vault}/terrawhisper/encyclopedia/csvs/systems-organs-ailments.csv'

ailments_rows = []
with open(systems_organs_ailments_filepath, newline='') as f:
    reader = csv.reader(f, delimiter='\\')
    for row in reader:
        ailments_rows.append(row)

for ailment_i, ailment_row in enumerate(ailments_rows[1:]):
    print(ailment_row)
    if ailment_row == []: continue

    system_slug = ailment_row[0]
    organ_slug = ailment_row[1]
    ailment_slug = ailment_row[2]
    ailment_name = ailment_row[3]

    ## for debug purpose (progression bar)
    with open(f'{vault}/terrawhisper/encyclopedia/csvs/ailments-keywords.txt') as f: content = f.read()
    ailments_to_process = [line.strip() for line in content.split('\n') if line.strip() != '']
    ailment_to_process_num = len(ailments_to_process)
    ailment_to_process_index = -1
    for tmp_i, ailment_to_process in enumerate(ailments_to_process):
        if ailment_to_process.lower().strip() == ailment_name.lower().strip():
            ailment_to_process_indes = tmp_i
            break

    ailment_filepath = f'{ailments_folderpath}/{ailment_slug}.json'
    if not os.path.exists(ailment_filepath): 
        with open(ailment_filepath, 'w') as f: f.write('{}')

    data = json_read(ailment_filepath)
    data['system'] = f'{system_slug.replace("-", " ")} system'.title()
    data['organ'] = f'{organ_slug.replace("-", " ")}'.title()

    # if 'causes' in data: del data['causes']
    # if 'symptoms' in data: del data['symptoms']
    # if 'remedies' in data: del data['remedies']
    # for remedy in data['remedies']:
        # if 'parts' in remedy['attributes']: del remedy['attributes']['parts']
    # json_write(ailment_filepath, data)
    # return

    key = 'definition'
    if key not in data:
        prompt = f'''
            Write a definition of the following ailment: {ailment_name}. 
            Reply in a 40 words paragraph.
            Start the reply with the following words: {ailment_name} is .
        '''
        reply = llm_reply(prompt, model, max_tokens=1024)
        print('***********************')
        print('***********************')
        print('***********************')
        print(f'{ailment_to_process_index}/{ailment_to_process_num}')
        print(reply)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = reply.strip()
        json_write(ailment_filepath, data)

    if 'causes' not in data or 'symptoms' not in data:
    # if 0:
        item_num = 20
        prompt = f'''
            Write 2 lists for the following ailment: {ailment_name}.
            In list 1, write the {item_num} most common causes.
            In list 2, write the {item_num} most common symptoms.
            Cause and symptoms must be different.
            Reply only with the names of the causes and symptoms, don't add descriptions.
            Reply in as few words as possible.
            Include only one cause or symptom per list item.
            Don't use brackets.
            Reply in the following JSON format:
            {{
                "causes": ["cause 1", "cause 2", "cause 3", "etc."],
                "symptoms": ["symptom 1", "symptom 2", "symptom 3", "etc."]
            }}
            Make sure you use double-quotes to surround each list item.
        '''
        causes = []
        symptoms = []
        for i in range(100):
            reply = llm_reply(prompt, model, max_tokens=256)
            try: reply_data = json.loads(reply.strip().lower())
            except: continue
            if 'causes' not in reply_data: continue
            if 'symptoms' not in reply_data: continue
            causes_list = reply_data['causes']
            symptoms_list = reply_data['symptoms']
            print(f'{ailment_to_process_index}/{ailment_to_process_num}')
            for line in causes_list:
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
            for line in symptoms_list:
                found = False
                for symptom in symptoms:
                    if line in symptom['name']: 
                        symptom['mentions'] += 1
                        found = True
                        break
                if not found:
                    symptoms.append({
                        'name': line, 
                        'mentions': 1, 
                    })
        causes = sorted(causes, key=lambda x: x['mentions'], reverse=True)
        symptoms = sorted(symptoms, key=lambda x: x['mentions'], reverse=True)
        print('***********************')
        print('***********************')
        print('***********************')
        for cause in causes[:item_num]:
            print(cause)
        print('***********************')
        print('***********************')
        print('***********************')
        print('***********************')
        print('***********************')
        print('***********************')
        for symptom in symptoms[:item_num]:
            print(symptom)
        print('***********************')
        print('***********************')
        print('***********************')
        data['causes'] = causes[:item_num]
        data['symptoms'] = symptoms[:item_num]
        json_write(ailment_filepath, data)

    key = 'preventions'
    if key not in data:
        prompt = f'''
            Write a paragraph about the main preventions of the following ailment: {ailment_name}. 
            Reply in a 40-60 words paragraph.
        '''
        reply = llm_reply(prompt, model, max_tokens=256)
        print('***********************')
        print('***********************')
        print('***********************')
        print(reply)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = reply.strip()
        json_write(ailment_filepath, data)

    key = 'complications'
    if key not in data:
        prompt = f'''
            Write a paragraph about the main complications of the following ailment if left untreated: {ailment_name}. 
            Reply in a 40-60 words paragraph.
        '''
        reply = llm_reply(prompt, model, max_tokens=256)
        print('***********************')
        print('***********************')
        print('***********************')
        print(f'{ailment_to_process_index}/{ailment_to_process_num}')
        print(reply)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = reply.strip()
        json_write(ailment_filepath, data)

    key = 'plants'
    if 'plants' not in data:
        ## retrieve documents
        query = f'herbs for {ailment_name}?'
        documents, metadatas = retrieve_docs(query)
        ## ask question
        question = f'''
            Write all the names of the medicinal plants that help with {ailment_name} mentioned in the following DOCUMENT. 
            Reply with only plants, not supplements.
            If the study doesn't mention herbs that help with {ailment_name}, reply only with: "None mentioned".
        '''
        output_plants = []
        for i, document in enumerate(documents):
            print('**************************************')
            print(f'{ailment_name} - {i}/{len(documents)}')
            print(f'{ailment_to_process_index}/{ailment_to_process_num}')
            print('**************************************')
            for i in range(10):
                prompt = f'''
                    {question}
                    Reply in numbered list format.
                    Don't hallucinate.
                    DOCUMENT: {document}
                '''
                reply = llm_reply(prompt, model, max_tokens=256)
                validator_reply = llm_validate(question, document, reply)
                try: validator_obj = json.loads(validator_reply)
                except: 
                    print('bad json')
                    continue
                score = validator_obj['SCORE']
                if score == 'PASS':
                    if 'none mentioned' in reply:
                        print('pass but none')
                        continue
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
                        prompt = f'''
                            Write the scientific name, also known as botanical name, of the following medicinal herb: {line}.
                            Reply only with the scientific name, don't add additional content.
                        '''
                        reply = llm_reply(prompt, model, max_tokens=256)
                        names_scientific = validate_names_scientific(reply)
                        for line in names_scientific:
                            found = False
                            for output_plant in output_plants:
                                if line in output_plant['plant_name_scientific']: 
                                    output_plant['plant_mentions'] += 1
                                    found = True
                                    break
                            if not found:
                                output_plants.append({
                                    'plant_name_scientific': line, 
                                    'plant_mentions': 1, 
                                })
                elif score == 'FAIL':
                    print('fail score')
                    continue
                else: 
                    print('bad score')
                    continue
                break
        output_plants = sorted(output_plants, key=lambda x: x['plant_mentions'], reverse=True)
        print('***********************')
        print('***********************')
        print('***********************')
        for output_plant in output_plants:
            print(output_plant)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = output_plants
        json_write(f'{jsons_folderpath}/{ailment_slug}.json', data)

    key = 'plants_desc'
    if key not in data:
        plants_names = [obj['plant_name_scientific'] for obj in data['plants']][:10]
        plants_names_str = ', '.join(plants_names)
        prompt = f'''
            Write a short paragraph on why the following plants help with the ailment {ailment_name}: {plants_names_str}. 
            Pack as much information as possible in as few words as possible.
            Write as few words as possible.
            Write less than 120 words.
            Write only proven facts, not opinions.
            Don't write fluff.
            Don't allucinate.
            Reply with a paragraph, don't write list.
            Start the reply with the following words: {plants_names[0]} .
        '''
        print(prompt)
        reply = llm_reply(prompt, model, max_tokens=1024)
        reply = reply.strip().replace('\n', ' ').replace('  ', ' ')
        print('***********************')
        print('***********************')
        print('***********************')
        print(f'{ailment_i}/{len(ailments_rows)}')
        print(f'{ailment_to_process_index}/{ailment_to_process_num}')
        print(reply)
        print('***********************')
        print('***********************')
        print('***********************')
        data[key] = reply.strip()
        json_write(ailment_filepath, data)

    ## remedies
    if 'remedies' not in data: data['remedies'] = []
    for plant_i, plant in enumerate(data['plants'][:3]):
        plant_name_scientific = plant['plant_name_scientific']

        found = False
        remedy_i = -1
        for i, remedy in enumerate(data['remedies']):
            remedy_plant_name = remedy['plant_name_scientific']
            if plant_name_scientific == remedy_plant_name:
                found = True
                remedy_i = i
                break

        if not found:
            data['remedies'].append({'plant_name_scientific': plant_name_scientific, 'attributes': {}})
            json_write(ailment_filepath, data)
        
        ## intro
        if 'intro' not in data['remedies'][plant_i]['attributes']:
            prompt = f'''
                Write a short 40 words paragraphs on why {plant_name_scientific} is used for {ailment_name} relief.
                Pack as much facts as possible in as few words as possible.
                Write only the 40 words paragraph, don't add additional information.
            '''
            reply = llm_reply(prompt, model, max_tokens=256)
            print('******************************************')
            print(f'{ailment_to_process_index}/{ailment_to_process_num}')
            print('******************************************')
            data['remedies'][plant_i]['attributes']['intro'] = reply
            json_write(ailment_filepath, data)

        ## constituents
        if 'constituents' not in data['remedies'][plant_i]['attributes']:
            prompt = f'''
                Write a list of the most important medicinal constituents of {plant_name_scientific} that help relieve {ailment_name}.
                Reply in the following JSON format:
                {{
                    "constituents": ["constituent 1", "constituent 2", "constituent 3", "etc."]
                }}
                Don't allucinate.
                Write only one constituent per list item.
                Write only the names of the constituents in the list items.
                Write as few words as possible.
                Make sure you write each constituent in between double-quotes.
            '''
            constituents = []
            for i in range(100):
                reply = llm_reply(prompt, model, max_tokens=256)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                print(f'{ailment_i}/{len(ailments_rows)}')
                print(f'{ailment_to_process_index}/{ailment_to_process_num}')
                print(ailment_name)
                print(plant_i)
                print(plant_name_scientific)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                try: reply_data = json.loads(reply.strip().lower())
                except: continue
                if 'constituents' not in reply_data: continue
                constituents_list = reply_data['constituents']
                for line in constituents_list:
                    line = line.strip('-')
                    found = False
                    for constituent in constituents:
                        if line in constituent['name']: 
                            constituent['mentions'] += 1
                            found = True
                            break
                    if not found:
                        constituents.append({
                            'name': line, 
                            'mentions': 1, 
                        })
            constituents = sorted(constituents, key=lambda x: x['mentions'], reverse=True)
            print('***********************')
            print('***********************')
            print('***********************')
            for constituent in constituents[:10]:
                print(constituent)
            print('***********************')
            print('***********************')
            print('***********************')
            data['remedies'][remedy_i]['attributes']['constituents'] = constituents[:20]
            json_write(ailment_filepath, data)

        if 'parts' not in data['remedies'][plant_i]['attributes']:
            prompt = f'''
                Write a list of the most important botanical parts of the plant {plant_name_scientific} that are used to relieve {ailment_name}.
                Reply in the following JSON format:
                {{
                    "parts": ["part 1", "part 2", "part 3", "etc."]
                }}
                Examples of plant parts are: leaf, root, flower, stem, etc.
                Don't allucinate.
                Write only one part per list item.
                Write only the names of the parts in the list items.
                Write as few words as possible.
                Make sure you write each part in between double-quotes.
            '''
            parts = []
            for i in range(20):
                reply = llm_reply(prompt, model, max_tokens=256)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                print(f'{ailment_i}/{len(ailments_rows)}')
                print(f'{ailment_to_process_index}/{ailment_to_process_num}')
                print(ailment_name)
                print(plant_i)
                print(plant_name_scientific)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                try: reply_data = json.loads(reply.strip().lower())
                except: continue
                if 'parts' not in reply_data: continue
                parts_list = reply_data['parts']
                for line in parts_list:
                    line = line.strip('-')
                    found = False
                    for part in parts:
                        if line in part['name']: 
                            part['mentions'] += 1
                            found = True
                            break
                    if not found:
                        parts.append({
                            'name': line, 
                            'mentions': 1, 
                        })
            parts = sorted(parts, key=lambda x: x['mentions'], reverse=True)
            print('***********************')
            print('***********************')
            print('***********************')
            for part in parts[:10]:
                print(part)
            print('***********************')
            print('***********************')
            print('***********************')
            data['remedies'][remedy_i]['attributes']['parts'] = parts[:20]
            json_write(ailment_filepath, data)

        if 'preparations' not in data['remedies'][plant_i]['attributes']:
            prompt = f'''
                Write a list of the most important medicinal herbal preparations of the plant {plant_name_scientific} that are used to relieve {ailment_name}.
                Reply in the following JSON format:
                {{
                    "preparations": ["preparation 1", "preparation 2", "preparation 3", "etc."]
                }}
                Examples of preparations are: tea, tincture, decoction, syrup, etc.
                Don't allucinate.
                Write only one preparation per list item.
                Write only the names of the preparations in the list items.
                Write as few words as possible.
                Make sure you write each preparation in between double-quotes.
            '''
            preparations = []
            for i in range(20):
                reply = llm_reply(prompt, model, max_tokens=256)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                print(f'{ailment_i}/{len(ailments_rows)}')
                print(f'{ailment_to_process_index}/{ailment_to_process_num}')
                print(ailment_name)
                print(plant_i)
                print(plant_name_scientific)
                print('******************************************')
                print('******************************************')
                print('******************************************')
                try: reply_data = json.loads(reply.strip().lower())
                except: continue
                if 'preparations' not in reply_data: continue
                preparations_list = reply_data['preparations']
                for line in preparations_list:
                    line = line.strip('-')
                    found = False
                    for preparation in preparations:
                        if line in preparation['name']: 
                            preparation['mentions'] += 1
                            found = True
                            break
                    if not found:
                        preparations.append({
                            'name': line, 
                            'mentions': 1, 
                        })
            preparations = sorted(preparations, key=lambda x: x['mentions'], reverse=True)
            print('***********************')
            print('***********************')
            print('***********************')
            for preparation in preparations[:10]:
                print(preparation)
            print('***********************')
            print('***********************')
            print('***********************')
            data['remedies'][remedy_i]['attributes']['preparations'] = preparations[:20]
            json_write(ailment_filepath, data)

        ## precautions
        if 'precautions' not in data['remedies'][plant_i]['attributes']:
            prompt = f'''
                Write a short 40 words paragraphs on the possible side effect of {plant_name_scientific} when used for {ailment_name} relief and the precautions to take.
                Pack as much facts as possible in as few words as possible.
                Write only the 40 words paragraph, don't add additional information.
            '''
            reply = llm_reply(prompt, model, max_tokens=256)
            data['remedies'][plant_i]['attributes']['precautions'] = reply
            json_write(ailment_filepath, data)




