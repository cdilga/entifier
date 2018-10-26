# The cols we want are 16, 17, 18 as they contain the normalised:
'''
16: normalised arg1
17: normalised rel
18: normalised arg2
'''

import pandas as pd
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
        self._load(filename)
        self._dbconnect('bolt://localhost:7687', 'neo4j', 'neo4')
        self._df = reader(filename)

    def _load(self, filename):
        #actually just use pandas here
        #do some parsing here 
        

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
        for row in self._df.iloc[:,[15,16,17]].iterrows():
            self.addNode(row[1][0], row[1][1], row[1][2])


    @staticmethod
    def call(tx, aname, relname, bname):
        result = tx.run("CREATE(a: Entity)-[re:Entity]->(b: Entity)" +
                        "SET a.name=$aname, b.name=$bname, re.name=$name", aname=aname, bname=bname, name=relname)
        return result

    def close(self):
        self._driver.close()


e = EntityLoader('data/reverb-output.csv', lambda x : pd.read_csv(x, sep='\t', lineterminator='\n'))
e.push()

