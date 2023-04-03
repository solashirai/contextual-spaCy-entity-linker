from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDFS
from yfiles_jupyter_graphs import GraphWidget
from collections import defaultdict


class LinkedEntityGraph:

    WD_NS = Namespace("http://www.wikidata.org/entity/")

    def __init__(self, core_entities, connecting_entities,
                 triples=[], uri2label=dict()):
        self.core_entities = set([URIRef(c) for c in core_entities])
        self.connecting_entities = set([URIRef(c) for c in connecting_entities])
        self.triples = triples
        self.graph = Graph()
        for (s,p,o) in self.triples:
            self.graph.add((s,p,o))

        self.uri2label = uri2label

    def get_yfiles_widget(self):
        # convert the graph into a format that's suitable to visaulize using the yfiles_jupyter_graphs package
        w = GraphWidget()

        # if something has a rank relation it is a statement, probably.
        wbrank = URIRef("http://wikiba.se/ontology#rank")
        statements = set()
        for ent in self.graph.all_nodes():
            has_outgoing = False
            for _ in self.graph.objects(subject=ent, predicate=wbrank):
                statements.add(ent)
                break

        nodelist = []
        edgelist = {}
        # for statements, the outgoing edge property is indicated by the following prefix
        statement_prefix = "http://www.wikidata.org/prop/statement/"
        # values with this prefix from wikidata are not particularly useful for now, so we'll ignore them
        wd_value_prefix = "http://www.wikidata.org/value"
        stp_len = len(statement_prefix)
        wdv_len = len(wd_value_prefix)
        for ent in statements:
            edge_label = str(ent)
            # if we see that a statement is only related to relevant entities because of a qualifier, mark it as bad
            # and don't add.
            # e.g. ("MIT", "official website", "web.mit.edu") qualified by "language"->"english"
            # where we're interested in "MIT" and "english" as nodes
            bad_statement = False

            thisedge = {"id": str(ent)}
            propdict = defaultdict(lambda: [])
            for p,o in self.graph.predicate_objects(subject=ent):
                if o[:wdv_len] == wd_value_prefix:
                    continue

                # this is the output of a statement
                if str(p)[:stp_len] == statement_prefix:
                    #if o not in self.core_entities and o not in self.connecting_entities:
                    #    bad_statement = True
                    #    break
                    thisedge["end"] = str(o)
                    edge_label = self.uri2label.get(LinkedEntityGraph.WD_NS[p[stp_len:]], str(p))
                else:
                    prop_id = str(p).split("/")[-1]
                    prop_lab = self.uri2label.get(LinkedEntityGraph.WD_NS[prop_id], str(p))
                    # there's some danger around here to accidentally link different properties with the same name
                    # but it's probably safe enough for wikidata... if not, it's probably a confusing property name.
                    propdict[prop_lab].append(o)
            if bad_statement:
                continue
            propdict = {str(k):v for k,v in propdict.items()}
            propdict['label'] = edge_label
            thisedge["properties"] = propdict

            edgelist[ent] = thisedge

        for ent in self.core_entities:
            thisnode = self.setup_node(ent, edgelist)
            nodelist.append(thisnode)
        for ent in self.connecting_entities:
            thisnode = self.setup_node(ent, edgelist)
            nodelist.append(thisnode)

        w.nodes = nodelist
        w.edges = list(edgelist.values())
        w.directed = True

        # scale mapping needs to be set separately instead of just setting it in the node
        scale_dict = {}
        for ent in self.core_entities:
            scale_dict[str(ent)] = 2
        for ent in self.connecting_entities:
            scale_dict[str(ent)] = 1
        def scale_mapping(index, node):
            return scale_dict[node["properties"]['wikidata URI']]

        w.set_node_scale_factor_mapping(scale_mapping)

        return w

    def setup_node(self, ent, edgelist):
        thisnode = {"id": str(ent)}
        propdict = defaultdict(lambda: [])
        for p, o in self.graph.predicate_objects(subject=ent):
            if o in edgelist.keys():
                edgelist[o]["start"] = str(ent)

        propdict = {str(k): v for k, v in propdict.items()}
        propdict['label'] = self.uri2label.get(ent, str(ent))
        propdict['wikidata URI'] = str(ent)
        thisnode["properties"] = propdict
        return thisnode