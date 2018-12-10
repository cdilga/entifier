from scipy.spatial import KDTree
import numpy as np
import sys
from dbloader import EntityLoader, yagoReader, csvRead, \
    convertToEmbeddings, loadGensim


class ShortestWord2VecDistanceClassifier:
  def __init__(self, threshold, target_words, target_embeddings):
    self.threshold = threshold
    self.target_words = target_words
    self.vec_tree = KDTree(target_embeddings)

  def closest_word(self, embeddings):
    distances, indices = self.vec_tree.query(embeddings)
    results = [yago_entities[i] if d < self.threshold else None
               for d, i in zip(distances, indices)]
    return results

  def closest_word_with_distance(self, embeddings):
    distances, indices = self.vec_tree.query(embeddings)
    results = [(yago_entities[i], d) if d < self.threshold else (None, d)
               for d, i in zip(distances, indices)]
    return results


# so that python doesn't explode
sys.setrecursionlimit(10000)


yago_obj = EntityLoader('data/yagoFacts.tsv', yagoReader)
yago_obj.cache()

wiki_obj = EntityLoader('data/wikipedia-full-reverb.txt', csvRead)
wiki_obj.cache()

yago = yago_obj._df
wiki = wiki_obj._df

gensim = loadGensim()

yago_entities = yago.iloc[:, 0].append(yago.iloc[:, 2]).unique()
yago_entity_embeddings = convertToEmbeddings(yago_entities, gensim)


# note we can make this 3 times faster by only calculating the mappings
# for unique wiki_entries (only 30% of the total)

wiki_entities_1 = wiki.iloc[:, 0]
wiki_entity_1_embeddings = convertToEmbeddings(wiki_entities_1, gensim)

wiki_entities_2 = wiki.iloc[:, 2]
wiki_entity_2_embeddings = convertToEmbeddings(wiki_entities_2, gensim)

model = ShortestWord2VecDistanceClassifier(threshold=1,
                                           target_words=yago_entities,
                                           target_embeddings=yago_entity_embeddings)

wiki['e1p'] = model.closest_word(wiki_entity_1_embeddings)
wiki['e2p'] = model.closest_word(wiki_entity_2_embeddings)

wiki.to_csv('wiki_pred.tsv', sep='\t', index=False)
