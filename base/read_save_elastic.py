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
        es = elasticsearch.Elasticsearch('http://127.0.0.1:9200/')

        # Iterate through the rows of the CSV file
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # row - список строк в файле, без названий столбцов
                internal_id, external_id, image_path, price, description = row
                embedding = self.from_img_to_vector(image_path)
                #                 embedding = embedding.tolist()

                # Save the data to Elasticsearch
                doc = {
                    'internal_id': internal_id,
                    'external_id': external_id,
                    'image_path': image_path,
                    'price': price,
                    'description': description,
                    'embedding': embedding
                }

                #                 es.indices.create(index='image_retrieval')
                es.index(index='image_retrieval', body=doc)
