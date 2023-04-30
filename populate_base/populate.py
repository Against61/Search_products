from elasticsearch import Elasticsearch

#internal imports
from config import config
from populate_base.read_save_elastic import ImageRetrieval, index_creation

es = Elasticsearch(config.ELASTICSEARCH_URL)

# Create an instance of Elasticsearch
if es.indices.exists(index=config.INDEX_NAME):
    es.update(index=config.INDEX_NAME, body=doc, doc_type='_doc')
    print("Индекс с именем", config.INDEX_NAME, "существует в Elasticsearch")
else:
    print("Индекс с именем", config.INDEX_NAME, "не существует в Elasticsearch")
    index_creation(config.INDEX_NAME, config.EMBEDDING_SIZE)
    # Create an instance of the ImageRetrieval class
    ir = ImageRetrieval(config.DATA_INDEX_DIR, device, val_transform)
    ir.save_to_elasticsearch()






