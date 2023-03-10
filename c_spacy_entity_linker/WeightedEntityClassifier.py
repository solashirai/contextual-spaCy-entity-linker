from itertools import groupby
import numpy as np
from typing import List, Tuple
from .EntityElement import EntityElement


class WeightedEntityClassifier:
    def __init__(self):
        pass

    def _get_grouped_by_length(self, entities):
        sorted_by_len = sorted(entities, key=lambda entity: len(entity.get_span()), reverse=True)

        entities_by_length = {}
        for length, group in groupby(sorted_by_len, lambda entity: len(entity.get_span())):
            entities = list(group)
            entities_by_length[length] = entities

        return entities_by_length

    def _filter_max_length(self, entities):
        entities_by_length = self._get_grouped_by_length(entities)
        max_length = max(list(entities_by_length.keys()))

        return entities_by_length[max_length]

    def _weight_max_prior(self, entities) -> List[Tuple[EntityElement, float]]:
        priors = [entity.get_prior() for entity in entities]
        max_prior = max(priors)
        # take the sqrt of prior/max so that entities with lower priors have a slightly better weighting
        # the lowest weighting possible is 0.5, while the highest is 1.0.
        return [(e, 0.5+0.5*(priors[ind]/max_prior)**0.5) for ind, e in enumerate(entities)]#entities[np.argmax(priors)]

    def _get_casing_difference(self, word1, original):
        difference = 0
        for w1, w2 in zip(word1, original):
            if w1 != w2:
                difference += 1

        return difference

    def _weight_most_similar(self, entities):
        similarities = np.array(
            [self._get_casing_difference(entity.get_span().text, entity.get_original_alias()) for entity in entities])

        min_indices = np.where(similarities == similarities.min())[0].tolist()

        return [entities[i] for i in min_indices]

    def __call__(self, entities) -> List[Tuple[EntityElement, float]]:
        filtered_by_length = self._filter_max_length(entities)

        return self._weight_max_prior(filtered_by_length)
