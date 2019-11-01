from elasticsearch import Elasticsearch
from tqdm import tqdm

def es_iterate_all_documents(es, index, pagesize=250, scroll_timeout="3m", **kwargs):
    """
    Helper to iterate ALL values from a single index
    Yields all the documents.
    """
    is_first = True
    while True:
        # Scroll next
        if is_first: # Initialize scroll
            result = es.search(index=index, scroll="1m", **kwargs, body={
                "size": pagesize
            })
            is_first = False
        else:
            result = es.scroll(body={
                "scroll_id": scroll_id,
                "scroll": scroll_timeout
            })
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]
        # Stop after no more docs
        if not hits:
            break
        # Yield each entry
        yield from (hit['_source'] for hit in hits)

es = Elasticsearch([{"host": "localhost"}])
with open("pubmed_dump.txt", "w+") as pubmed_writer:
    for entry in tqdm(es_iterate_all_documents(es, 'pubmed'), total=2.5e6):
        #dict_keys(['full_title', 'abstract', 'journal', 'pmid', 'pmc', 'doi', 'publication_year', 'all_text', 'subjects'])
        line = f"{entry['full_title'].strip()}\n{entry['abstract'].strip()}\n{entry['all_text'].strip()}"
        line = line.strip().replace('\n\n', '\n')
        pubmed_writer.write(line+'\n\n') #Empty line for document seperation for BERT
