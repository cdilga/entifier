# The cols we want are 16, 17, 18 as they contain the normalised:
'''
16: normalised arg1
17: normalised rel
18: normalised arg2
'''

import pandas as pd
import numpy as np
import urllib
from neo4j.v1 import GraphDatabase

class EntityLoader():
    '''
    This object should be able to load a file of entity relations
    from reverb and load into neo4j. We will try to make it slightly 
    general, but really this should just be something our pipeline can call 
    on a file and this takes care of the rest.
    Single use.
    '''

    def __init__(self, filename, reader):
        self._dbconnect('bolt://localhost:7687', 'neo4j', 'neo4')
        self._df = reader(filename)

    def _dbconnect(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def getModel(self):
        '''Returns the internal df'''
        return(self._df)
    
    def addNode(self, arg1, rel, arg2):
        with self._driver.session() as session:
            response = session.write_transaction(EntityLoader.call, arg1, rel, arg2)
            
    def push(self):
        #print(self._df)
        for row in self._df.iloc[:,[0,1,2]].iterrows():
            self.addNode(row[1][0], row[1][1], row[1][2])


    @staticmethod
    def call(tx, aname, relname, bname):
        result = tx.run("MERGE (a: Entity {{ name: '{}' }}) MERGE (b: Entity {{ name: '{}' }}) MERGE (a)-[re:`{}`]->(b)".format(
            urllib.parse.quote(relname.strip(), safe=' '),
            urllib.parse.quote(aname.strip(), safe=' '),
            urllib.parse.quote(bname.strip(), safe=' ')))
        #tx.run("MATCH (a) MERGE (a {{a.name}})")
        return result

    def close(self):
        self._driver.close()

def yagoReader(filename):
    arr = []
    with open(filename, encoding="utf8") as f:
        for line in f:
            arr.append([el.replace('<', '').replace('>', '') for el in line.strip().split('\t')])
    return pd.DataFrame(arr)

def csvRead(filename):
    df = pd.read_csv(filename, sep='\t', lineterminator='\n').iloc[:, [15, 16, 17]]

    return df


e = EntityLoader('data/yagoFacts.tsv', yagoReader)
#e = EntityLoader('data/reverb-output.csv', csvRead)
e.push()

