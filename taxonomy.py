import csv
import json

from oliark_llm import llm_reply

vault = '/home/ubuntu/vault'
llms_path = f'{vault}/llms'
model = f'{llms_path}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf' 

ailments_filepath = f'{vault}/terrawhisper/encyclopedia/csvs/ailments-keywords.txt'
with open(ailments_filepath) as f: content = f.read()

systems_organs_ailments_filepath = f'{vault}/terrawhisper/encyclopedia/csvs/systems-organs-ailments.csv'

n_tries = 100
for ailment_name in content.split('\n'):
    ailment_name = ailment_name.lower().strip()
    if ailment_name == '': continue

    rows = []
    with open(systems_organs_ailments_filepath, newline='') as f:
        reader = csv.reader(f, delimiter='\\')
        for row in reader:
            rows.append(row)

    found = False
    for row in rows[1:]:
        print(row)
        if row == []: continue

        csv_ailment_name = row[3]
        if ailment_name == csv_ailment_name:
            found = True
            break

    if not found:
        prompt = f'''
            Write the body system and body part most relevant to the following ailment: {ailment_name}.
            The possible of body systems are: 
            - circulatory 
            - digestive
            - endocrine
            - integumentary
            - lymphatic
            - muscular
            - nervous
            - reproductive
            - respiratory
            - skeletal
            - urinary
            Example of body part is: heart, leg, brain, etc.
            Reply with the following JSON format: 
            {{
                "system": <system name>,
                "part": <body part>,
            }}
            Reply only with the json, don't add additional information.
        '''
        systems = []
        parts = []
        for i in range(n_tries):
            reply = llm_reply(prompt)
            try: 
                data = json.loads(reply)
            except:
                continue
            system_name = data['system'].lower().strip()
            part_name = data['part'].lower().strip()

            found = False
            for system in systems:
                if system_name in system['name']: 
                    system['mentions'] += 1
                    found = True
                    break
            if not found:
                systems.append({
                    'name': system_name, 
                    'mentions': 1, 
                })

            found = False
            for part in parts:
                if part_name in part['name']: 
                    part['mentions'] += 1
                    found = True
                    break
            if not found:
                parts.append({
                    'name': part_name, 
                    'mentions': 1, 
                })

        systems = sorted(systems, key=lambda x: x['mentions'], reverse=True)
        parts = sorted(parts, key=lambda x: x['mentions'], reverse=True)

        print('***********************')
        for system in systems:
            print(system)
        for part in parts:
            print(part)
        print('***********************')
        
        ailment_slug = ailment_name.replace(' ', '-').lower().strip()
        with open(systems_organs_ailments_filepath, 'a') as f:
            writer = csv.writer(f, delimiter='\\')
            writer.writerow([systems[0]['name'], parts[0]['name'], ailment_slug, ailment_name])
