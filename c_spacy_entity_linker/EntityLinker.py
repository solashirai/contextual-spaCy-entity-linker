from spacy.tokens import Doc, Span
from spacy.language import Language
import requests
import random
import time

from ortools.sat.python import cp_model
from collections import defaultdict
from rdflib import URIRef, Literal

from .WeightedEntityClassifier import WeightedEntityClassifier
from .EntityCollection import EntityCollection
from .LinkedEntityGraph import LinkedEntityGraph
from .TermCandidateExtractor import TermCandidateExtractor

@Language.factory('entityLinker')
class EntityLinker:

    def __init__(self, nlp, name, query_url='https://query.wikidata.org/sparql',
                 direct_link_weight=300,
                 twohop_link_weight=200,
                 related_link_weight=100,
                 do_two_hops=False):
        Doc.set_extension("linkedEntities", default=EntityCollection(), force=True)
        Span.set_extension("linkedEntities", default=None, force=True)
        self.query_url = query_url
        self.direct_link_weight = direct_link_weight
        self.twohop_link_weight = twohop_link_weight
        self.related_link_weight = related_link_weight
        self.do_two_hops = do_two_hops

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

        stime = time.time()
        context_triples = self.run_construct_queries(all_entity_candidates=distinct_uris)
        print("finished running construct queries, ", time.time()-stime)
        ent_to_trip = defaultdict(lambda: set())
        for (s,p,o) in context_triples:
            ent_to_trip[s].add((s,p,o))
            ent_to_trip[o].add((s,p,o))
        stime = time.time()
        optimized_candidates, core_entities, linking_entities = self.lp_disambiguate(candidate_spans, context_triples)
        print('finished LP solving, ', time.time()-stime)

        for span, entity in optimized_candidates.items():
            # Add the entity to the sentence-level EntityCollection
            span.sent._.linkedEntities.append(entity)
            # Also associate the token span with the entity
            span._.linkedEntities = entity
            # And finally append to the document-level collection
            entities.append(entity)

        relevant_triples, uri2label = self.collect_relevant_subgraph(core_entities=core_entities,
                                                                     linking_entities=linking_entities)
        leg = LinkedEntityGraph(core_entities, linking_entities, relevant_triples, uri2label)
        doc._.linkedEntities = EntityCollection(entities, leg)

        return doc

    def run_construct_queries(self, all_entity_candidates):
        if len(all_entity_candidates) == 0:
            return []

        queries_conducted = 0
        result_triples = []
        url = self.query_url

        # simplify the query to use "wd" prefix to reduce data sent
        # possibly TODO, break up the query based on how many entities there are
        # 2-hop starts to become too time consuming for constraint solving and/or querying.
        # filter to remove statements
        query_str = f"""
        CONSTRUCT {{
        ?s ?p ?o.
        {'?o ?pp ?oo .' if self.do_two_hops else ''}
        }}
        WHERE {{
            VALUES ?s {{ {' '.join([f'wd:{wid.split("/")[-1]}' for wid in all_entity_candidates])} }} .
            ?s ?p ?o .
            FILTER NOT EXISTS {{ ?s wikibase:rank ?r }}
            FILTER (!isLiteral(?o))
            {
            '''
            ?o ?pp ?oo .
            FILTER NOT EXISTS {{ ?o wikibase:rank ?r }}
            FILTER (!isLiteral(?oo))
            '''
            if self.do_two_hops else ''
            }
        }}
        """
        print("querying...")
        print(query_str)

        res_get = requests.get(url, params={'format':'json',
                                            'query': query_str})
        data = res_get.json()
        for row in data['results']['bindings']:
            # ignore bnodes and literals
            if row['object']['type'] == 'uri':
                result_triples.append((row['subject']['value'],row['predicate']['value'],row['object']['value']))

        return result_triples

    def get_entity_labels(self, entities):
        if len(entities) == 0:
            return {}
        url = self.query_url

        query_str = f"""
        SELECT ?item ?itemLabel
        WHERE {{
            VALUES ?item {{ {' '.join([f'wd:{wid.split("/")[-1]}' for wid in entities])} }} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        res_get = requests.get(url, params={'format':'json',
                                            'query': query_str})
        data = res_get.json()
        out_dict = {}
        for row in data['results']['bindings']:
            out_dict[URIRef(row['item']['value'])] = row['itemLabel']['value']
        return out_dict

    def collect_relevant_subgraph(self, core_entities, linking_entities):
        if len(core_entities) == 0:
            return {}, {}
        result_triples = []
        url = self.query_url
        uris_to_label = set()

        # simplify the query to use "wd" prefix to reduce data sent
        # possibly TODO, break up the query based on how many entities there are
        # we want to collect statements between entities
        # all statements have a rank, I think. beyond this there doesn't seem to be any easy way to identify
        # whether something is a statement or not... (e.g. no (x, type, wikidata:statement)).
        # we also split the process into 3 queries: core-core, core-link, and link-core
        for (from_ents, to_ents) in [(core_entities, linking_entities),
                                     (core_entities, core_entities),
                                     (linking_entities, core_entities)]:
            stime = time.time()
            query_str = f"""
            CONSTRUCT {{
                ?scon ?ps ?st .
                ?st wikibase:rank ?r ;
                  ?po ?sto ;
                  ?pcon ?conn .
            }}
            WHERE {{
                VALUES ?scon {{ {' '.join([f'wd:{wid.split("/")[-1]}' for wid in from_ents])} }} .
                VALUES ?conn {{ {' '.join([f'wd:{wid.split("/")[-1]}' for wid in to_ents])} }} .
                ?scon ?p ?conn .
                ?scon ?ps ?st .
                ?st wikibase:rank ?r ; 
                  ?po ?sto .
                FILTER EXISTS {{ ?st ?ppp ?conn }}
            }}
            """
            print("querying...")
            print(query_str)

            res_get = requests.get(url, params={'format':'json',
                                                'query': query_str})
            data = res_get.json()
            print("query time: ", time.time()-stime)
            wikidata_prop_pref = "http://www.wikidata.org/prop/"
            wikidata_entity = "http://www.wikidata.org/entity/Q" # wikidata entities start with Q ids
            for row in data['results']['bindings']:
                subj = URIRef(row['subject']['value']) if row['subject']['type'] == 'uri' else Literal(row['subject']['value'])
                obj = URIRef(row['object']['value']) if row['object']['type'] == 'uri' else Literal(row['object']['value'])
                pred = URIRef(row['predicate']['value'])
                result_triples.append((subj, pred, obj))

                # statements have a bunch of dashes in them. wikibase ontology terms have # in them.
                if isinstance(subj, URIRef):
                    if str(subj)[:len(wikidata_entity)] == wikidata_entity:
                        uris_to_label.add(subj)
                if isinstance(obj, URIRef):
                    if str(obj)[:len(wikidata_entity)] == wikidata_entity:
                        uris_to_label.add(obj)
                # predicate labels should be available from the wd prefix.
                # only do this for properties that come from wikidata
                if str(pred)[:len(wikidata_prop_pref)] == wikidata_prop_pref:
                    prop_id = pred.split("/")[-1]
                    uris_to_label.add(LinkedEntityGraph.WD_NS[prop_id])

        uri2label = self.get_entity_labels(uris_to_label)

        return result_triples, uri2label

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
            print(span, "-----")
            for (ent, weight) in mappings.get(span, []):
                print(ent.get_uri(), weight)
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
            objective_terms.append(sum([100*v*prior_weight[v] for v in nounchunk_vars]))

        # set up a dict "graph" first to make it easier to determine whether entities have direct or 1-hop
        # connections to each other
        g = defaultdict(lambda: set())
        for (s, p, o) in triples:
            g[s].add(o)

        dc_sets = defaultdict(lambda: set())
        hop_sets = defaultdict(lambda: set())
        dhop_sets = defaultdict(lambda: set())
        for ent in varsync.keys():
            for ent2 in varsync.keys():
                if ent == ent2 or ent in dc_sets[ent2] or ent in dhop_sets[ent2]:
                    continue
                if ent2 in g[ent]:
                    dc_sets[ent].add(ent2)
                else:
                    # only add to hopset once.
                    # dhop has higher score, so we'll still check to see if it's in the hop_sets, in which case
                    # we'll have to replace it
                    if ent in dhop_sets[ent2]:
                        continue
                    for o in g[ent]:
                        # ent->o->ent2 or ent->o<-ent2
                        if ent2 in g[o]:
                            dhop_sets[ent].add(ent2)
                            # remove the entry from hop_set if this has a dhop match, which is higher prio
                            if ent in hop_sets[ent2]:
                                hop_sets[ent2].remove(ent)
                            if ent2 in hop_sets[ent]:
                                hop_sets[ent].remove(ent2)
                            break
                        elif o in g[ent2]:
                            # safety check to ensure we only add to hop_set from one side
                            if ent not in hop_sets[ent2]:
                                hop_sets[ent].add(ent2)

        # set up objectives to increase score if a selected mapping has a relevant triples.
        # only give higher scores for unique subject-object pairs
        intermediate_score_vars = defaultdict(lambda: [])
        seen_tups = set()
        tup_vars = {}
        for ent in varsync.keys():

            #TODO add restriction like the following for if an intermediate hop entity is selected
            # going through another entity as a "common" entity is not particularly useful, since
            # it'll bias choices towards entities with lots of connections
            # if o in varsync.keys():
            #     continue

            for dc_ent in dc_sets[ent]:
                subj_vars = varsync[ent]
                obj_vars = varsync[dc_ent]

                t = model.NewIntVar(0,1,'')
                model.Add(t <= sum(subj_vars))
                model.Add(t <= sum(obj_vars))

                objective_terms.append(self.direct_link_weight*t)

            for hop_ent in dhop_sets[ent]:
                subj_vars = varsync[ent]
                obj_vars = varsync[hop_ent]

                t = model.NewIntVar(0,1,'')
                model.Add(t <= sum(subj_vars))
                model.Add(t <= sum(obj_vars))

                objective_terms.append(self.twohop_link_weight*t)

            for hop_ent in hop_sets[ent]:
                subj_vars = varsync[ent]
                obj_vars = varsync[hop_ent]

                t = model.NewIntVar(0,1,'')
                model.Add(t <= sum(subj_vars))
                model.Add(t <= sum(obj_vars))

                objective_terms.append(self.related_link_weight*t)

        model.Maximize(sum(objective_terms))
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return None
        else:
            mapping_choices = {}
            core_entities = set()
            linking_entities = set()
            for span in mappings.keys():
                for ent in nc_namemap[span]:
                    if solver.Value(nc_varmap[span][ent]) == 1:
                        mapping_choices[span] = uri2obj[ent]
                        core_entities.add(ent)

            # retrieve connected hop entities
            entlist = list(core_entities)
            for ind, ent in enumerate(entlist):
                for ent2 in entlist[ind+1:]:
                    if ent == ent2:
                        continue
                    if ent2 in g[ent] or ent in g[ent2]:
                        continue
                    else:
                        link_choices = []
                        for o in g[ent]:
                            if o in g[ent2] or ent2 in g[o]:
                                link_choices.append(o)
                        if ent == "http://www.wikidata.org/entity/Q640506":
                            print("????????", link_choices)
                        if link_choices:
                            # using too many linking entities will cause errors when querying wikidata
                            # randomly choose them for now, TODO choose based on prior or something
                            linking_entities.add(random.choice(link_choices))

            for k,vals in intermediate_score_vars.items():
                if sum([solver.Value(v) for v in vals]) >= 2:
                    linking_entities.add(k)

            # if any of the core entities also happens to be a linking entity, prioritize it as core
            linking_entities -= core_entities
            return mapping_choices, core_entities, linking_entities

    def lp_disambiguate_prev(self, mappings, triples):
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

            # objective_var = model.NewIntVar(0, 10, '')
            # model.Add(objective_var <= sum(indicators))

            objective_terms.append(self.hop_weight*objective_var)

        model.Maximize(sum(objective_terms))
        status = solver.Solve(model)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            return None
        else:
            mapping_choices = {}
            core_entities = set()
            linking_entities = set()
            for span in mappings.keys():
                for ent in nc_namemap[span]:
                    if solver.Value(nc_varmap[span][ent]) == 1:
                        mapping_choices[span] = uri2obj[ent]
                        core_entities.add(ent)
            for k,vals in intermediate_score_vars.items():
                if sum([solver.Value(v) for v in vals]) >= 2:
                    linking_entities.add(k)


            # if any of the core entities also happens to be a linking entity, prioritize it as core
            linking_entities -= core_entities
            return mapping_choices, core_entities, linking_entities
