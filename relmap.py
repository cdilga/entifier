import json
from dbloader import EntityLoader, csvRead, yagoReader
import numpy as np

with open('./data/entity_map_2.json', 'rb') as fp:
    entity_map = json.loads(fp.read())


def lookup_entity_map(entity):
    if entity is not None:
        return entity_map[entity]
    else:
        return None


yago_obj = EntityLoader('data/yagoFacts.tsv', yagoReader)
yago_obj.cache()

wiki_obj = EntityLoader('data/wikipedia-full-reverb.txt', csvRead)
wiki_obj.cache()

yago = yago_obj._df
wiki = wiki_obj._df

wiki = wiki.dropna()

wiki['e1p'] = wiki['e1'].apply(lookup_entity_map)
wiki['e2p'] = wiki['e2'].apply(lookup_entity_map)

wiki_no_none = wiki[wiki['e1p'].notnull() & wiki['e2p'].notnull()][['e1p', 'rel', 'e2p']]

df_merge = wiki_no_none.merge(yago, left_on=['e1p', 'e2p'], right_on=['e1', 'e2'])

df_merge[['rel_x', 'rel_y']].drop_duplicates().to_csv('merged_no_dupes.tsv', index=False, sep='\t')
