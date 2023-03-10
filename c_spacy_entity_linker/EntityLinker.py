from spacy.tokens import Doc, Span
from spacy.language import Language
import requests

from ortools.sat.python import cp_model
from collections import defaultdict

from .WeightedEntityClassifier import WeightedEntityClassifier
from .EntityCollection import EntityCollection
from .TermCandidateExtractor import TermCandidateExtractor

@Language.factory('entityLinker')
class EntityLinker:

    def __init__(self, nlp, name, query_url='https://query.wikidata.org/sparql', link_weight=100, hop_weight=100):
        Doc.set_extension("linkedEntities", default=EntityCollection(), force=True)
        Span.set_extension("linkedEntities", default=None, force=True)
        self.query_url = query_url
        self.link_weight = link_weight
        self.hop_weight = hop_weight

    def __call__(self, doc):
        tce = TermCandidateExtractor(doc)
        classifier = WeightedEntityClassifier()

        for sent in doc.sents:
            sent._.linkedEntities = EntityCollection([])

        candidate_spans = {}

        entities = []
        for termCandidates in tce:
            entityCandidates = termCandidates.get_entity_candidates()
            if len(entityCandidates) > 0:
                entity_weights = classifier(entityCandidates)
                # just use the first (entity, weight) tuple to get this entity's span
                span = doc[entity_weights[0][0].span_info.start:entity_weights[0][0].span_info.end]
                candidate_spans[span] = entity_weights

        remove_spans = set()
        for span, entity_weights in candidate_spans.items():
            entity = entity_weights[0][0]
            s_start = entity.span_info.start
            s_end = entity.span_info.end
            # if this span's start and end index are within another span's start and end,
            # choose the longer span for making our match.
            for other_span, other_ent_weights in candidate_spans.items():
                other_ent = other_ent_weights[0][0]
                if other_span == span:
                    continue
                o_start = other_ent.span_info.start
                o_end = other_ent.span_info.end
                if s_start >= o_start and s_end <= o_end:
                    remove_spans.add(span)
                    break
        for span in remove_spans:
            candidate_spans.pop(span)

        distinct_uris = set()
        for span, entity_weights in candidate_spans.items():
            for (e,w) in entity_weights:
                distinct_uris.add(e.get_uri())

        context_triples = self.run_construct_queries(all_entity_candidates=distinct_uris)
        optimized_candidates = self.lp_disambiguate(candidate_spans, context_triples)

        for span, entity in optimized_candidates.items():
            # Add the entity to the sentence-level EntityCollection
            span.sent._.linkedEntities.append(entity)
            # Also associate the token span with the entity
            span._.linkedEntities = entity
            # And finally append to the document-level collection
            entities.append(entity)

        doc._.linkedEntities = EntityCollection(entities)

        return doc

    def run_construct_queries(self, all_entity_candidates):
        queries_conducted = 0
        result_triples = []
        url = self.query_url

        # simplify the query to use "wd" prefix to reduce data sent
        # possibly TODO, break up the query based on how many entities there are
        query_str = f"""
        CONSTRUCT {{?s ?p ?o}}
        WHERE {{
            VALUES ?s {{ {' '.join([f'wd:{wid.split("/")[-1]}' for wid in all_entity_candidates])} }} .
            ?s ?p ?o .
        }}
        """

        res_get = requests.get(url, params={'format':'json',
                                            'query': query_str})
        data = res_get.json()
        for row in data['results']['bindings']:
            # ignore bnodes and literals
            if row['object']['type'] == 'uri':
                result_triples.append((row['subject']['value'],row['predicate']['value'],row['object']['value']))

        return result_triples

    def lp_disambiguate(self, mappings, triples):
        uri2obj = {}
        model = cp_model.CpModel()

        solver = cp_model.CpSolver()
        objective_terms = []

        # integers are scaled up to enable integer programming.
        # selecting entities is scaled by 10000, with an additional int score based on
        # the rank of a search entity to help break ties.
        # selecting triples is scaled by 100, to help overcome ties while being less important
        # than selecting more entities.

        nc_namemap = defaultdict(lambda: [])
        nc_varmap = defaultdict(lambda: {})
        prior_weight = {}
        varsync = defaultdict(lambda: [])
        # set up vars that indicate which WD entity a span is mapped to
        for span in mappings.keys():
            nounchunk_vars = []
            for (ent, weight) in mappings.get(span, []):
                var = model.NewIntVar(0,1,"")
                mapname = ent.get_uri()
                uri2obj[mapname] = ent

                prior_weight[var] = weight

                varsync[mapname].append(var)

                nc_namemap[span].append(mapname)
                nc_varmap[span][mapname] = var
                nounchunk_vars.append(var)
            if not nounchunk_vars:
                continue
            # only 1 mapping can be selected for this noun
            model.Add(sum(nounchunk_vars) <= 1)
            # this objective term will help to prefer mapping choices with better regular search rank
            objective_terms.append(sum([1000*v*prior_weight[v] for v in nounchunk_vars]))

        # set up objectives to increase score if a selected mapping has a relevant triples.
        # only give higher scores for unique subject-object pairs
        intermediate_score_vars = defaultdict(lambda: [])
        seen_tups = set()
        tup_vars = {}
        for trip in triples:
            if (trip[0], trip[2]) in seen_tups:
                continue
            else:
                seen_tups.add((trip[0], trip[2]))

            # we're looking at a 1hop context around entities, so there are two cases for us to consider.
            # first, the triple directly connects two relevant entities.
            # second, the triple connects to an entity that is connected to another relevant entity.

            # case 1
            if trip[0] in varsync and trip[2] in varsync:
                # ortools won't let us just multiply variables together, so use this as a workaround
                subj_vars = varsync[trip[0]]
                obj_vars = varsync[trip[2]]

                t = model.NewIntVar(0,1,'')
                tup_vars[(trip[0], trip[2])] = t
                model.Add(t <= sum(subj_vars))
                model.Add(t <= sum(obj_vars))

                objective_terms.append(self.link_weight*t)

            # case 2 will always be set up since it might occur even when case 1 is also true
            # in the current setup trip[0] should always be in varsync
            subj_vars = varsync[trip[0]]
            indicator_var = model.NewIntVar(0,1,'')
            model.Add(indicator_var <= sum(subj_vars))
            intermediate_score_vars[trip[2]].append(indicator_var)

        # add solutions to check that the intermediate triples are supported from both ends, and add to
        # the objective terms appropriately
        for int_ent, indicators in intermediate_score_vars.items():
            indicator_bool = model.NewBoolVar('')
            objective_var = model.NewIntVar(0,1,'')
            # the objective var will be 2 iff at least 2 indicators =1
            model.Add(sum(indicators) >= 2).OnlyEnforceIf(indicator_bool)
            model.Add(objective_var <= 0).OnlyEnforceIf(indicator_bool.Not())

            objective_terms.append(self.hop_weight*objective_var)

        model.Maximize(sum(objective_terms))
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return None
        else:
            mapping_choices = {}
            for span in mappings.keys():
                for ent in nc_namemap[span]:
                    if solver.Value(nc_varmap[span][ent]) == 1:
                        mapping_choices[span] = uri2obj[ent]
            return mapping_choices
