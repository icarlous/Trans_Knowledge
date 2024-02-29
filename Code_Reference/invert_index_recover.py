"""
This script read inverted index of abstracts and translate them back to plain text and calculate their Specter2 embeddings
Author: Shiyang Lai
Date: Feb 3 2024
"""
import os
import ast
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

tokenizer = AutoTokenizer.from_pretrained('/project/jevans/shiyang/scripts/science/specter2')
model = AutoAdapterModel.from_pretrained('/project/jevans/shiyang/scripts/science/specter2')
model.load_adapter("/project/jevans/shiyang/scripts/science/specter2")
model.set_active_adapters('specter2')

parser = argparse.ArgumentParser(description='Specify the starting point and chunk size.')
parser.add_argument('-start_line', type=int, help='The line number to start processing from.')
parser.add_argument('-chunk_size', type=int, help='The number of lines to process each time.')
parser.add_argument('-end_size', type=int, help='The line number to end processing from.')
args = parser.parse_args()

FULL_SIZE = args.end_size
CHUNK_SIZE = args.chunk_size
START_LINE = args.start_line

input_file_path = "/project/jevans/donghyun/Assembly_Tree/id_title_abstract_joined_for_sharing/Neuroscience.csv"
# output_file_path = "/project/jevans/donghyun/OpenAlex_Dec_20_2023/csv-filed_work_only_abstract/abstract_specter2_embedding.csv"
output_file_path = f'/project/jevans/shiyang/science/oa-article-embeddings/neuro_science/chunk_{START_LINE}.csv'

def compute_embedding(papers):
    try:
        text_batch = [d['title'] + tokenizer.sep_token + d['abstract'] for d in papers]
    except:
        raise TypeError(f"{papers[0]['title']} -- {papers[0]['abstract']}")
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

def process_chunk(start_line, chunk_size, input_file, output_file):
    try:
        df = pd.read_csv(input_file, skiprows=start_line, nrows=chunk_size, header=None if start_line > 0 else 'infer')
    except:
        try:
            df = pd.read_csv(input_file, skiprows=start_line, header=None if start_line > 0 else 'infer')
        except:
            # raise ValueError('Cannot read the file!')
            print(f'ERROR! ----- {input_file}')
            return
    df.columns = ['id', 'concept_id', 'concept_display_name', 'score', 'title', 'publication_year', 'type', 'abstract_inverted_index']
    df['embedding'] = None
        
    for index, row in df.iterrows():
        if row['embedding'] != None:
            continue
        work_abstract = recover_plain_text(row['abstract_inverted_index'])
        paper_embedding = compute_embedding([{'title': str(row['title']), 'abstract': str(work_abstract)}])
        df.loc[index, 'embedding'] = np.array2string(paper_embedding[0].detach().numpy(), precision=5, separator=',')
    df.to_csv(output_file, mode='a', header=False, index=False, sep=';')

def recover_plain_text(input_string):
    try:
        input_string = re.sub(r'<.*?>', '', input_string)
        word_index = []
        _ = ast.literal_eval(input_string)
        for key, value in _.items():
            for index in value:
                word_index.append([key, index])
        word_index = sorted(word_index, key = lambda x : x[1])
        plain_text = ' '.join([word[0] for word in word_index])
    except:
        plain_text = ''
    return plain_text


if os.path.exists(output_file_path):
    os.remove(output_file_path)
    print(f"The file {output_file_path} has been removed.")
else:
    print(f"The file {output_file_path} does not exist.")


with tqdm(total=FULL_SIZE-START_LINE) as pbar:
    for i in range(START_LINE, FULL_SIZE, CHUNK_SIZE):
        process_chunk(start_line=i, chunk_size=CHUNK_SIZE, input_file=input_file_path, output_file=output_file_path)
        pbar.update(CHUNK_SIZE)