import regex
import subprocess
import codecs
import lxml.etree as etree
from tqdm import tqdm
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

from urllib.request import urlretrieve
from os.path import isfile, isdir, getsize
def download_dump(arch_uri="https://dumps.wikimedia.org/plwiki/latest/", 
                  file="plwiki-latest-pages-articles-multistream.xml.bz2 "):
    datafile="{}".format(file)
    if not (isfile(datafile) or isfile(datafile[:-4])):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=file) as pbar:
            urlretrieve(arch_uri+file, datafile, pbar.hook)
    print ("Downloading DONE")
 
    return datafile


def unbzip2(filepath):
    bashCommand = ["bzip2",'-d', filepath]
    try:
        output = subprocess.check_output(bashCommand, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as pserror: 
        print (pserror.output)
    else:
        print ("DONE {}".format(output))
    return filepath[:-4]


def clean_text(text, lcode):
    
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    if lcode in ['ko']: # korean
        text = regex.sub(u"[^ \r\n\p{Hangul}.?!]", " ", text) # Replace unacceptable characters with a space.
    elif lcode in ['ja']: # japanese
        text = regex.sub(u"[^\r\n\p{Han}\p{Hiragana}\p{Katakana}ー。！？]", "", text)
    elif lcode in ['zh']: # chinsese
        text = regex.sub(u"[^\r\n\p{Han}。！？]", "", text)
    elif lcode in ['th']: # thai
        text = regex.sub(u"[^ \r\n\p{Thai}.?!]", " ", text)
    elif lcode in ['ru']: # russian
        text = regex.sub(u"[^ \r\n\p{Cyrillic}.?!\-]", " ", text)
        text = text.lower()
#     elif lcode in ['ar']: # arabic
#         text = regex.sub(u"[^ \r\n\p{Arabic}.?!\-]", " ", text)
    elif lcode in ['hi']: # hindi
        text = regex.sub(u"[^ \r\n\p{Devanagari}.।?!\-]", " ", text)
    elif lcode in ['bn']: # bengali
        text = regex.sub(u"[^ \r\n\p{Bengali}.।?!\-]", " ", text)
    elif lcode in ['de']: # german
        text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
    else: # Mostly european languages
        text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
        text = text.lower()
    
    # Common
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    return text

def sentence_segment(text, lcode):
    '''
    Args:
      text: A string. A unsegmented paragraph.
    
    Returns:
      A list of sentences.
    '''
    if lcode in ['ja', 'zh']:
        sents = regex.split(u"([。！？])?[\n]+|[。！？]", text) 
    elif lcode in ['th']:
        sents = text.split("[\n]+") 
    elif lcode in ['hi', 'bn']: # hindi, bengali
        sents = regex.split(u"([.।?!])?[\n]+|[.।?!] ", text)
    elif lcode in ['de']: # german
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
        sents = [sent[0].lower() + sent[1:] for sent in sents if sent is not None and len(sent) > 1]
    else:
        sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    return sents
        
def word_segment(sent, lcode):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A list of words.
    '''
    if lcode in ['ko']:
        words = [word for word, _ in kkma.pos(sent)]
    elif lcode in ['ja']:
        words = mecab.parse(sent.encode('utf8')).split() 
    elif lcode in ['th']:
        words = pythai.split(sent)
    elif lcode in ['vi']:
        words = ViTokenizer.tokenize(sent).split()        
    elif lcode in ['zh']:
        words = list(jieba.cut(sent, cut_all=False)) 
#     elif lcode in ['ar']:
#         words = segmenter.segment(sent).split()
    else: # Mostly european languages
        words = sent.split()
    
    return words

def build_corpus(filepath, max_corpus_size=100000000000, lcode="pl"):
    txt_file=datafile="{}.txt".format(filepath[:-4])
    print(txt_file)
 
    with codecs.open(txt_file, 'w', 'utf-8') as fout:
        i = 1
        j = 1
        ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace
        for _, elem in tqdm(etree.iterparse("plwiki-latest-pages-articles-multistream.xml.bz2 .out", tag=ns+"text")):
            running_text = elem.text
            try:
                running_text = clean_text(running_text, lcode)
                sents = sentence_segment(running_text, lcode)
                for sent in sents:
                    if sent is not None:
                        words = word_segment(sent, lcode)
                        if len(words) > 10:
                            if lcode in ['ja']:
                                fout.write(" ".join(words).decode('utf8') + "\n")
                            else:
                                fout.write(" ".join(words) + "\n")
                                
            except:
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!
            if i % 1000 == 0: 
                #print ('.', end='')
                fsize = os.path.getsize(txt_file)
                if fsize > max_corpus_size:
                    break
            i += 1
    print ("DONE")