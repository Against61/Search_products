import read_save_elastic
from utils.utils import read_image

def search(query_image, top_k=10):
    # Search for the most similar images
    res = es.search(index='image_retrieval', body={
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'image_embedding') + 1.0",
                    "params": {"query_vector": query_embedding[0].tolist()}
                }
            }
        }
    })
    # Return the results
    return res['hits']['hits']

query_image = 'C:/cache/torchok/data/sop/Stanford_Online_Products/sofa_final/110891955769_1.JPG'
query_embedding = ir.from_img_to_vector(query_image)
query_embedding

result = search(query_embedding, top_k=10)
result[0]['_source']['path']

#show query image and finding image


figure, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

for i in range(10):
    ax.ravel()[0].imshow(read_image(query_image))
    ax.ravel()[0].set_axis_off()
    ax.ravel()[0].set_title('Query image')
    ax.ravel()[i].imshow(read_image(result[i]['_source']['path']))
    ax.ravel()[i].set_axis_off()
    ax.ravel()[i].set_title(f"Similar image {i+1}")
    plt.tight_layout()
plt.show()

