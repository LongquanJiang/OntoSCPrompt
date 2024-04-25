import json
from .args import *
import urllib
from typing import List
from SPARQLWrapper import SPARQLWrapper, JSON

def execute_query(endpoint: str, query: str) -> List[str]:

    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)

    try:
        response = sparql.query().convert()
    except urllib.error.URLError as e:
        print(e)
        print(query)
        exit(0)

    if "boolean" in response:  # ASK
        results = [response["boolean"]]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]:  # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                for k, v in res.items():
                    results.append(v["value"])
    return results

if __name__ == '__main__':
    query = "select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 }"

    kb_endpoint = "https://query.wikidata.org/sparql"

    results = execute_query(kb_endpoint, query)

    print(results)