# Search_products
Search for similar products from different sources

This solution is still under development. Only one module has been implemented so far, with 4

At the moment, only the training part is functioning, to produce a model that generates the imaging embeddings. 

What parts the solution consists of:

1. the training part to look for the similarity is done on the basis of the representation task. As an example, Dataloader is set up to train on an SOP dataset.  Therefore it is necessary to download this dataset, specifying the path for the training and validation data in Dataloader.py. 

2. data indexing. The training model will be used to retrieve the embeddings and save the images to the elastic search engine. Where ANN will be used and nearest cosine distance search -- in progress

3. parsing module. Is configured directly to the source of parsing, but forms a file under the received format elastic search --in progress

4. Results display module. Allows you to configure the nearest neighbours from different previously parsed sources --in progress


Additional modules:

1. Method for validating the accuracy of a similarity search.

2. Re-indexing with only updated data

3. method of similar by synthetics texts to get additional similarity properties of products

Translated with www.DeepL.com/Translator (free version)
