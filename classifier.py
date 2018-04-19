import os
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def get_relative_path(path):
    filenames = []
    filepaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root,file)
            filepaths.append(filepath)
            filenames.append(file)
    return filenames,filepaths


def read_urls(path):
    filenames = get_relative_path(path)
    url_lists = []
    for filename in filenames:
        urls = open(filename).readlines()
        for url in urls:
            url = url.strip()
            url_lists.append(url)
    return url_lists


def read_docs(path):
    filenames,filepaths = get_relative_path(path)
    docs = []
    doc_names = []
    stopwrds = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    for filepath,filename in zip(filepaths,filenames):
        f = open(filepath, 'r', encoding='utf-8')
        doc = ''
        temp =  f.readlines()
        for line in temp:
            doc += line.lower()
        doc = nltk.word_tokenize(doc)
        doc = [token for token in doc if token not in stopwrds]
        docs.append(doc)
        doc_names.append(filename)
    return doc_names, docs


def index_doc(docs,doc_names):
    index = {}
    inverse_index = {}
    for doc,doc_name in zip(docs,doc_names):
        word_count={}
        for word in doc:
            if word in word_count.keys():
                word_count[word]+=1
            else:
                word_count[word]=1
        index[doc_name] = word_count # 각 다큐먼트당 언급되 단어 수 dictionary
        #index[doc_name]=nltk.Text(doc).vocab()
    for doc in index.keys():
        doc_index = index[doc]
        for word in doc_index.keys():
            if word in inverse_index.keys():
                inverse_index[word].append(doc)
            else:
                inverse_index[word] = [doc] # 단어별로 언급된 다큐먼트 나열
    return index, inverse_index


def build_dictionary(index):
    dictionary = {}
    for word in index.keys():
        dictionary[word]=len(dictionary)
    return dictionary # 전체 다큐먼트의 모든 단어들 별로 사용된 횟수 저장


def compute_tfidf(index,word_dictionary,doc_dictionary):
    vocab_size = len(word_dictionary)
    doc_size = len(doc_dictionary)
    tf = np.zeros((doc_size,vocab_size))
    for doc in index:
        index_per_doc = index[doc]
        vector = np.zeros(vocab_size)
        for word in index_per_doc:
            vector[word_dictionary[word]] = index_per_doc[word]
        vector = np.log(vector+1)
        tf[doc_dictionary[doc]] = vector
    idf_numerator = doc_size
    idf_denominator = np.sum(np.sign(tf), 0)
    idf = np.log(idf_numerator/idf_denominator)
    tfidf = tf*idf
    return tfidf


def cosine_similarity(x,y):
    normalizing_factor_x = np.sqrt(np.sum(np.square(x)))
    normalizing_factor_y = np.sqrt(np.sum(np.square(y)))
    return np.matmul(x,np.transpose(y))/(normalizing_factor_x*normalizing_factor_y)


def query_matching(inverse_dictionary,query):
    set_list = [set(inverse_dictionary[word]) for word in query]
    return set.union(*set_list)


def query_tf_idf(word_dictionary, query):
    vocab_size = len(word_dictionary)
    tf = np.zeros(vocab_size)

    for word in query:
        tf[word_dictionary[word]] += 1
    tf = np.log(tf + 1)

    idf_numerator = tfidf.shape[1]
    idf_denominator = np.sum(np.sign(tfidf), 0)
    idf = np.log(idf_numerator / idf_denominator)
    return tf * idf


if __name__ == '__main__':
    doc_names, docs = read_docs('data')
    index, inverted_index = index_doc(docs,doc_names) # 각 다큐먼트당 언급된 단어 수 dictionary,
    word_dictionary = build_dictionary(inverted_index) # 총 단어 개수 dictionary
    doc_dictionary = build_dictionary(index)
    tfidf = compute_tfidf(index,word_dictionary,doc_dictionary) # shape : (60,6198)

    stopwrds = stopwords.words('english')
    f = open('./input_document.txt', 'r', encoding='utf-8')
    input_doc = '' # test document
    temp = f.readlines()

    for line in temp:
        input_doc += line.lower()
    input_doc = nltk.word_tokenize(input_doc)
    input_doc = [token for token in input_doc if token not in stopwrds]

    result = ''
    for containing_token in input_doc:
        result = inverted_index[containing_token]

    result = list(set(result)) # input 다큐먼트의 단어가 포함된 다큐먼트의 목록

    scores = {}
    query_vector = query_tf_idf(word_dictionary, input_doc)
    # print(query_vector)

    for doc in result:
        doc_index = doc_dictionary[doc]
        scores[doc] = cosine_similarity(query_vector, tfidf[doc_index])

    label = ''
    max_val = 0
    for key, value in scores.items():
        if value > max_val:
            label = key
            max_val = value

    print(label)
