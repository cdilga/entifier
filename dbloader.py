
'''
The cols we want are 16, 17, 18 as they contain the normalised:

16: normalised arg1
17: normalised rel
18: normalised arg2
'''

import pandas as pd
import numpy as np
import urllib
from neo4j.v1 import GraphDatabase
import string

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))
import pickle
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

class EntityLoader():
    '''
    This object should be able to load a file of entity relations
    from reverb and load into neo4j. We will try to make it slightly 
    general, but really this should just be something our pipeline can call 
    on a file and this takes care of the rest.
    Single use.
    '''

    def __init__(self, filename, reader):
        self._filename = filename
        
        try:
            self._df = pd.read_csv(self._filename[:-4] + '.cache.txt', sep=',', encoding='utf8')
        except:
            self._df = reader(filename)

    def _dbconnect(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def getModel(self):
        '''Returns the internal df'''
        return(self._df)
    
    def addNode(self, arg1, rel, arg2):
        with self._driver.session() as session:
            try:
                response = session.write_transaction(EntityLoader.call, arg1, rel, arg2)
            except neo4j.exceptions.ClientError as e:
                print(e)
            
    def push(self):
        self._dbconnect('bolt://localhost:7687', 'neo4j', 'neo4')

        for row in self._df.iloc[:,[0,1,2]].iterrows():
            self.addNode(row[1][0], row[1][1], row[1][2])
        self.close()

    def cache(self):
        self._df.to_csv(self._filename[:-4] + '.cache.txt', sep=',', encoding='utf8', index=False)   

    @staticmethod
    def call(tx, aname, relname, bname):
        result = tx.run("MERGE (a: ReverbEntity {{ name: '{}' }}) MERGE (b: ReverbEntity {{ name: '{}' }}) MERGE (a)-[re:`{}`]->(b)".format(
            urllib.parse.quote(str(aname).strip(), safe=' '),
            urllib.parse.quote(str(bname).strip(), safe=' '),
            urllib.parse.quote(str(relname).strip(), safe=' ')))
        #tx.run("MATCH (a) MERGE (a {{a.name}})")
        print('.', end='')
        return result

    def close(self):
        self._driver.close()

def yagoReader(filename):
    arr = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            arr.append([el.replace('<', '').replace('>', '').lower() for el in line.strip().split('\t')])
    return pd.DataFrame(arr, columns=['e1', 'rel', 'e2'])

def csvRead(filename):
    #These functions for reading could use improvement
    df = pd.read_csv(filename, sep='\t', lineterminator='\n',
                     encoding='utf8',dtype=str, error_bad_lines=False, header=None).iloc[:, [15, 16, 17]]
    df = df.rename(columns={15: "e1",
                       16: "rel", 
                       17: "e2"})
    translation = {' ': '_'}
    for char in string.punctuation:
        translation[char] = None
    
    for s in df.columns:
        df[s] = df[s].apply(lambda x: ' '.join([word for word in word_tokenize(
            str(x)) if word not in stop_words]).translate(str.maketrans(translation)).strip())
        
    
    print('Successfully read in file')
    return df


yago = EntityLoader('data/yagoFacts.tsv', yagoReader)
#yago.cache()

wiki = EntityLoader('data/wikipedia-full-reverb.txt', csvRead)
#df = csvRead('data/wikipedia-partial-output.txt')
#print(wiki._df.head())
#wiki.cache()
#wiki.push()

def compare(entity1, entity2):
    '''Take two dataframes with 3 columns and compare entities'''
    temp1 = pd.concat([entity1.iloc[:,0], entity1.iloc[:,2]])
    temp2 = pd.concat([entity2.iloc[:, 0], entity2.iloc[:, 2]])
    result = temp1.isin(temp2)
    return result

def findMatches(entity1, entity2):

    #indicies = 
    merged = pd.merge(entity1, entity2, left_on=['e1', 'e2'], right_on=['e1', 'e2'])
    merge2 = pd.merge(entity1, entity2, left_on=[
                      'e1', 'e2'], right_on=['e2', 'e1'])
    #merged = entity1.values[entity1.iloc[:, 0].isin(entity2.iloc[:, 0])]
    return pd.concat([merged, merge2])

#print(compare(wiki._df, yago._df).value_counts())
#findMatches(wiki._df, yago._df).to_csv(open('matches.csv', 'w'))

def loadGensim():
    try:
        gensimmodel = pickle.load(open('gensim.cache', 'rb'))
        print('Loaded gensim from cache')
    except:
        from gensim.models import KeyedVectors
        print('Fallback to loading gensim from source')
        #NOTE You can swap these comments with the middle gensim line
        #gensimmodel = KeyedVectors.load_word2vec_format('./google-news/GoogleNews-vectors-negative300.bin', binary=True, limit=50000)
        gensimmodel = KeyedVectors.load_word2vec_format('./google-news/GoogleNews-vectors-negative300.bin', binary=True)
        #pickle.dump(gensimmodel, open('gensim.cache', 'wb'))
        print('Loaded Gensim')
    
    return gensimmodel
class KillClassifier():
    def __init__(self, threshold):
        self._threshold = threshold

    def fit(self, x, y):
        #for the first element 
        x = np.array(x)
        y = np.array(y)
        self.train_x = x
        self.train_y = y

    def matrix_cosine(self, x, y):
        return np.einsum('ij,ij->i', x, y) / (
            np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )
    def predict(self, x):
        if(len(x) == 0):
            raise AttributeError('passed length of 0 which is dumb???')

        x = np.array(x)
        for i, k in enumerate(x):
            ret = np.chararray((x.shape[0], 1), unicode = True)
            #temp = np.empty_like(x)
            temp = self.matrix_cosine(np.broadcast_to(x[i], self.train_x.shape), self.train_x)
            if not np.isnan(temp[0]):
                index = np.nanargmax(temp)
                if temp[index] > self._threshold:
                    ret[i] = self.train_y[index]
                else:
                    ret[i] = None
            else:
                ret[i] = None
        return ret

def lookup(gensim, k):
    try:
        return gensim[k]
    except:
        return np.zeros_like(gensim['the'])

def convertToEmbeddings(words, gensim):
    ret = []
    for i, val in enumerate(words):
        ret.append(
            np.mean(np.array([lookup(gensim, k) for k in str(val).split('_')]), axis=0))
    ret = np.array(ret)
    if np.all(np.isnan(np.array(ret))):
        raise ValueError

    return ret

def classifyEntities(wiki, yago):
    '''Maps the wiki classifications to yago ones, or declares them 'out of class' '''
    gensim = loadGensim()
    #will use the yago vecs inside a knn that is essentially pretrained....
    #essentially an argmin of the distances to known classes
    #make the word vecs of the unique yago classes the training x and the actual words the labels
    clr = KillClassifier(0.8)
    
    #fakeGensim = {'the': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

    training_y = yago.iloc[:, 0].append(yago.iloc[:, 2])
    
    print(training_y.shape)
    training_x = convertToEmbeddings(training_y, gensim)
    if training_x.shape[0] != training_y.shape[0]:
        raise ValueError('Somehow the x training shape is wrong')
    print('Finished converting embeddings')
    #then find the closest by some distance metric. Then go for it.
    #print(training_x)
    #print(training_x)

    clr.fit(training_x, training_y)
    print('Finished training model')
    #first convert wiki to word2vec embeddings 
    #run first column through predict

    #run second column through predict
    finalDf = pd.DataFrame()
    print('e1 shape: {}'.format(wiki['e1'].values.shape))
    e1Embeddings = convertToEmbeddings(wiki['e1'], gensim)
    print('e1 Embeddings calculated')
    if e1Embeddings.size == 0:
        print("ERROR")
    finalDf['e1w'] = wiki['e1']
    finalDf['e1p'] = clr.predict(e1Embeddings)
    #finalDf['e1y'] = yago['e1']
    print('predicted e1')
    finalDf['rel'] = wiki['rel']
    e2Embeddings = convertToEmbeddings(wiki['e2'], gensim)

    print('e2 Embeddings calculated')
    if e2Embeddings.size == 0:
        print("ERROR")
    finalDf['e2w'] = wiki['e2']
    finalDf['e2p'] = clr.predict(e2Embeddings)
    #finalDf['e2y'] = yago['e2']
    print('predicted e2')

    
    return finalDf

    #if it's further than some distance then we just leave it out of class

#NOTE you can replace the first : with a :1000 to get a smallersubset
deambiguated = classifyEntities(wiki._df.iloc[:, :], yago._df.iloc[:, :])
saveDf = deambiguated.loc[np.logical_and(deambiguated['e1p'] != '', deambiguated['e2p'] != '')]
saveDf.to_csv(open('fullDataDump.csv', 'w'))

