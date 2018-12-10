from scipy.spatial import KDTree
import os
import pickle
import time
from dbloader import EntityLoader, yagoReader, csvRead, \
    convertToEmbeddings, loadGensim


# jumping through all these hoops to avoid loading gensim into memory
# unless we absolutely have to

def load_or_pickle(pickle_file, load_func, *args):
    if pickle_file in os.listdir():
        with open(pickle_file, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(pickle_file, 'wb') as f:
            obj = load_func(*args)
            pickle.dump(obj, f)
    return obj


def load_yago_entities(yago):
    return yago.iloc[:, 0].append(yago.iloc[:, 2]).unique()


def load_embeddings(entities):
    gensim = loadGensim()
    return convertToEmbeddings(entities, gensim)


def get_wiki_entities(wiki, index):
    return wiki.iloc[:, index]


class ShortestWord2VecDistanceClassifier:
    def __init__(self, threshold, target_words, target_embeddings):
        self.threshold = threshold
        self.target_words = target_words
        self.vec_tree = KDTree(target_embeddings)

    def closest_word(self, embeddings):
        distances, indices = self.vec_tree.query(embeddings)
        results = [self.target_words[i] if d < self.threshold else None
                   for d, i in zip(distances, indices)]
        return results

    def closest_word_with_distance(self, embeddings):
        distances, indices = self.vec_tree.query(embeddings)
        results = [(self.target_words[i], d) if d < self.threshold else (None, d)
                   for d, i in zip(distances, indices)]
        return results

    def closest_word_single(self, embedding):
        distance, index = self.vec_tree.query(embedding)
        if distance < self.threshold:
            return self.target_words[index]
        else:
            return None


yago_obj = EntityLoader('data/yagoFacts.tsv', yagoReader)
yago_obj.cache()

wiki_obj = EntityLoader('data/wikipedia-full-reverb.txt', csvRead)
wiki_obj.cache()

yago = yago_obj._df
wiki = wiki_obj._df

yago_entities = load_or_pickle('yago_entities.pickle', load_yago_entities, yago)
yago_entity_embeddings = load_or_pickle('yago_entity_embeddings.pickle', load_embeddings, yago_entities)

wiki_entities_1 = load_or_pickle('wiki_entities_1.pickle', get_wiki_entities, wiki, 0)
wiki_entity_1_embeddings = load_or_pickle('wiki_entity_1_embeddings.pickle', load_embeddings, wiki_entities_1)

wiki_entities_2 = load_or_pickle('wiki_entities_2.pickle', get_wiki_entities, wiki, 2)
wiki_entity_2_embeddings = load_or_pickle('wiki_entity_2_embeddings.pickle', load_embeddings, wiki_entities_2)

model = ShortestWord2VecDistanceClassifier(threshold=1,
                                           target_words=yago_entities,
                                           target_embeddings=yago_entity_embeddings)


def wiki_unique_entitiy_map(wiki_entities_1, wiki_entities_2):
    wiki_entities_unique = list(wiki_entities_1.unique()) + list(wiki_entities_2.unique())
    wiki_embeddings_unique = load_embeddings(wiki_entities_unique)
    unique_entity_map = {ent: (emb, '<UNK>') for ent, emb in zip(wiki_entities_unique, wiki_embeddings_unique)}
    return unique_entity_map


if __file__ == '__main__':

    wiki_unique_entitiy_map = load_or_pickle('wiki_unique_entitiy_map.pickle',
                                             wiki_unique_entitiy_map,
                                             wiki_entities_1, wiki_entities_2)

    start = time.time()
    i = 0
    completed = 0
    chk = int(len(wiki_unique_entitiy_map) / 100)
    for entity in wiki_unique_entitiy_map:
        i += 1
        embedding, target_class = wiki_unique_entitiy_map[entity]
        # don't recalculate data
        if target_class != '<UNK>':
            pass
        else:
            target_class = model.closest_word_single(embedding)
            wiki_unique_entitiy_map[entity] = (embedding, target_class)

        # make a checkpoint every 1 %
        if i % chk == 0:
            end = time.time()
            print("Checkpoint: {}, took {}".format(i, end - start))
            with open('wiki_unique_entitiy_map.pickle', 'wb') as f:
                pickle.dump(wiki_unique_entitiy_map, f)
            start = end

    with open('wiki_unique_entitiy_map.pickle', 'wb') as f:
        pickle.dump(wiki_unique_entitiy_map, f)

    entity_map = {entity: wiki_unique_entitiy_map[entity][1] for entity in wiki_unique_entitiy_map}
    with open('entity_map.json', 'w') as fp:
        json.dump(entity_map, fp)
