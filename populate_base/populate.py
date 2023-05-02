from elasticsearch import Elasticsearch



#internal imports
from config import config
from train.augmentation import val_transform
from populate_base.read_save_elastic import ImageRetrieval, index_creation

es = Elasticsearch(config.ELASTICSEARCH_URL)
device = config.DEVICE

# Create an instance of Elasticsearch
if es.indices.exists(index=config.INDEX_NAME):
    es.update(index=config.INDEX_NAME, body=doc, doc_type='_doc')
    print("Индекс с именем", config.INDEX_NAME, "существует в Elasticsearch")
    logger.info("Индекс с именем %s существует в Elasticsearch", config.INDEX_NAME)
else:
    print("Индекс с именем", config.INDEX_NAME, "не существует в Elasticsearch")
    logger.info("Индекс с именем %s не существует в Elasticsearch", config.INDEX_NAME)
    index_creation(config.INDEX_NAME, config.EMBEDDING_SIZE)
    logger.info("Создаем индекс с именем %s", config.INDEX_NAME)
    # Create an instance of the ImageRetrieval class
    ir = ImageRetrieval(config.DATA_INDEX_DIR, device, val_transform)
    ir.save_to_elasticsearch()






