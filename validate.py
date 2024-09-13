from oliark import file_read
from oliark_llm import llm_reply

vault = f'/home/ubuntu/vault'
llms_folderpath = f'{vault}/llms'
model_generator_filepath = f'{llms_folderpath}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
model_validator_filepath = f'{llms_folderpath}/Llama-3-Partonus-Lynx-8B-Intstruct-Q4_K_M.gguf'


def generate(question, document_filepath):
    context = file_read(document_filepath)
    prompt = f'''
        Reply to the following QUESTION using the following DOCUMENT:
        QUESTION: {question}
        DOCUMENT: {context}
    '''
    answer = llm_reply(prompt, model_generator_filepath)
    return answer


def validate(question, document_filepath, answer):
    context = file_read(document_filepath)
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
    reply = llm_reply(prompt, model_validator_filepath)
    return reply


question = 'who are the protagonists of final fantasy 7?'
document_filepath = 'context.txt'

answer = generate(question, document_filepath)
reply = validate(question, document_filepath, answer)
