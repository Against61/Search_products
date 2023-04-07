import cv2
import elasticsearch
import numpy as np
import csv
import json


class ImageRetrieval:
    def __init__(self, csv_file, model, device, val_transform):
        self.csv_file = csv_file
        self.model = model
        self.device = device
        self.val_transform = val_transform

    def from_img_to_vector(self, query_image):
        image = cv2.imread(str(query_image), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms and move on device
        query_image = self.val_transform(image=image)
        query_image = query_image['image'].unsqueeze(0).to(self.device)

        # Get image embedding
        query_embedding = self.model.forward(query_image).detach().cpu().numpy()

        return query_embedding

    def save_to_elasticsearch(self):
        # Connect to Elasticsearch
        es = elasticsearch.Elasticsearch('http://localhost:9200')

        # Iterate through the rows of the CSV file
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # row - список строк в файле, без названий столбцов
                image_id, class_id, image_path, path = row
                path = 'C:/cache/torchok/data/sop/Stanford_Online_Products/' + path
                embedding = self.from_img_to_vector(path)
                image_id = int(image_id)
                class_id = int(class_id)
                # Save the data to Elasticsearch
                doc = {
                    'image_id': image_id,
                    'class_id': class_id,
                    'super_class_id': image_path,
                    'path': path,
                    'image_embedding': embedding[0].tolist(),  # Преобразовываем ndarray в список
                    # 'doc_type': 'doc' # add the doc_type key
                }

                es.index(index='image_retrieval', body=doc, doc_type='_doc')

# Create an instance of the ImageRetrieval class
ir = ImageRetrieval("C:/Users/Alexei/output_file.csv", device, val_transform)

#put to elastic search dense vector
res = es.indices.create(index='image_retrieval', body=
{
    "mappings": {
        "properties": {
            "image_path": {
                "type": "keyword"
            },
            "price": {
                "type": "float"
            },
            "description": {
                "type": "text"
            },
            "image_embedding": {
                "type": "dense_vector",
                "dims": 768
            }
        }
    }
})

ir.save_to_elasticsearch()
