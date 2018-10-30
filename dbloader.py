
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
    return pd.DataFrame(arr)

def csvRead(filename):
    #These functions for reading could use improvement
    df = pd.read_csv(filename, sep='\t', lineterminator='\n',
                     encoding='utf8',dtype=str, error_bad_lines=False).iloc[:, [15, 16, 17]]
    
    translation = {' ': '_'}
    for char in string.punctuation:
        translation[char] = None
    
    for s in df.columns:
        df[s] = df[s].apply(lambda x: str(x).strip().translate(str.maketrans(translation)))
        
    
    print('Successfully read in file')
    return df


#yago = EntityLoader('data/yagoFacts.tsv', yagoReader)
#yago.cache()

wiki = EntityLoader('data/wikipedia-partial-output.txt', csvRead)
wiki.cache()
wiki.push()
def compare(entity1, entity2):
    '''Take two dataframes with 3 columns and compare entities'''
    temp1 = pd.concat([entity1.iloc[:,0], entity1.iloc[:,2]])
    temp2 = pd.concat([entity2.iloc[:, 0], entity2.iloc[:, 2]])
    result = temp1.isin(temp2)
    return result

#print(compare(wiki._df, yago._df).value_counts())

