import regex
from tqdm import tqdm
from collections import Counter

def display_text(file_path, till_row=10):
    with open(file_path, "r") as f:
        for i, sent in enumerate(f):
            if i > till_row-1:
                break
            print(sent)
            
            
def clean_text(file_path, new_file_path, total_line):
    punc_repl_dict={    
    '.': 'PERIOD',
    ',': 'COMMA',
    '!': 'EXCLA',
    '?': 'QUEST',
    '"': 'QUOTA',
    ';': 'SEMICOLON',
    '(': 'L_PAREN',
    ')': 'R_PAREN',
    '-': 'DASH_'
    }
    reg_exp = regex.compile("|".join(map(regex.escape, punc_repl_dict.keys())))
    wordcounts = Counter()
    with open(file_path, "r") as f, open(new_file_path, "w+") as g:
        for line in tqdm(f, total=total_line):
            line = line.lower()
            line = regex.sub(r'^ - ', '', line)
            line = regex.sub(r'[^ \n\p{Latin}]', ' ', line)
#             line = regex.sub(r'(.)(\p{P})', r'\1 \2', line)
#             line = regex.sub(r'(\p{P})([^\s])', r'\1 \2', line)
#             line = regex.sub(r'\p{P}', ' ', line)
#             line = reg_exp.sub(lambda match: punc_repl_dict[match.group(0)], line)
            line = regex.sub(r' +', ' ', line)
            g.write(line)
            wordcounts.update(line.split())
    return wordcounts
        
        
def save_word2vec_tsv_format(model, file_name):
    with open("{}_tensors.tsv".format(file_name), 'w') as tensors:
        with open("{}_metadata.tsv".format(file_name), 'w') as metadata:
            for word in tqdm(model.wv.index2word):
                metadata.write(word + '\n')
                vector_row = '\t'.join(map(str, model[word]))
                tensors.write(vector_row + '\n')
                

