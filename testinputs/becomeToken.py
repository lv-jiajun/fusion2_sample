import os

from gensim import models
from gensim.models import Word2Vec


# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':' , ';',
    '{', '}'
    }


def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 3])
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i + 2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))


class MyCorpus(object):
    """ Data Preparation \n
    gensim’s word2vec expects a sequence of sentences as its input. Each sentence is a list of words (utf8 strings).

    Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything
    in RAM: we can provide one sentence, process it, forget it, load another sentence...

    Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract
    named entities… All of this can be done inside the MySentences iterator and word2vec doesn’t need to know. All that is
    required is that the input yields one sentence (list of utf8 words) after another.
    """

    def __init__(self, dirname, suffix):
        self.dirname = dirname
        self.suffix = suffix

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith(self.suffix):  # filter irrelevant files
                for line in open(os.path.join(self.dirname, fname)):
                    if not line.startswith(">>>"):  # is not summary line
                        yield line.split()



def trainWord2Vec(corpus_path, dic_file_path, suffix, save_whole_model=True):
    """
    obtain a phaseII dictionary with skip-gram model
    :param corpus_path:
    :param dic_file_path:
    :param suffix:
    :param save_whole_model: default True, save the whole model. otherwise just save the standalone keyed vectors
    :return:
    """
    texts = MyCorpus(corpus_path, suffix)
    #model = Word2Vec(texts, size=100, window=5, min_count=5, workers=4, sg=1)
    model = models.FastText(texts, size=100, window=5, min_count=5,  sg=0, min_n=2, max_n=3)

   # if save_whole_model:
    model.save(dic_file_path)
    model.wv.save_word2vec_format('word2vec.vector')
   # else:
  #      model.wv.save(dic_file_path)


def main():
    '''
    sum = 0
    filename = 'D:/JiajunLv/word2vec_cfg/sellector.csv'
    file_new = open('D:/JiajunLv/word2vec_cfg/new2CFG#coarse.csv', 'w', encoding='utf-8')
    with open(filename, "r+", encoding="utf8") as file:
        for line in file:
            if line.startswith(">>>cfg&"):
                sum = sum + 1

            stripped = line.strip()
            if not stripped:  # 判断是否是空，如果是则continue
                continue
            elif stripped.startswith("**************************"):
                sum = sum+1
                continue
            elif stripped.startswith(">>>func&"):
                file_new.write(line)
            elif stripped.startswith(">>>cfg&"):
                file_new.write(line)
            else:
                line = line[1:-2]
                line1 = tokenize(line)
                file_new.write(str(line1))
                file_new.write('\n')

    print(sum)
    file.close()
'''
    corpus_path = 'E:/图神经网络/testinputs'
    dic_file_path = 'E:/图神经网络/testinputs/vec4/ins2vec_coarse.dic'
    suffix ='newCFG#coarse.csv'
    trainWord2Vec(corpus_path, dic_file_path, suffix)




if __name__ == "__main__":
    main()