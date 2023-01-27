"""
This script allows you to create a new collection if it does not exist and to conduct a text similarity search.
"""
import time
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from typing import List
from constants import EmbeddingModelConstants, DatasetConstants, MilvusServerConstants, CollectionConstants, IndexConstants, SearchConstants
import pandas as pd
from tqdm import tqdm
from loguru import logger
import argparse


def connect_to_server():
    """It creates a Milvus connection.

    """
    logger.info("start connecting to Milvus")
    connections.connect(alias="default", host=MilvusServerConstants.HOST, port=MilvusServerConstants.PORT)


def create_collection(collection_name) -> Collection:
    """It creates a new collection with the specified schema

    :return: The created collection
    """
    fields = [
        FieldSchema(name=CollectionConstants.ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True,
                    max_length=100),

        FieldSchema(name=CollectionConstants.TEXT_FIELD_NAME, dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name=CollectionConstants.EMBEDDINGS_FIELD_NAME, dtype=DataType.FLOAT_VECTOR,
                    dim=EmbeddingModelConstants.DIM)
    ]

    schema = CollectionSchema(fields, collection_name)
    logger.info("Collection {} created".format(collection_name))
    embedding_collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")
    create_index(embedding_collection)
    return embedding_collection


def create_index(embedding_collection):
    """It builds an index on the embeddings field

    :param embedding_collection: The collection
    """

    logger.info("Indexes {} created".format(IndexConstants.INDEX_TYPE))
    index_parameters = {
        "index_type": IndexConstants.INDEX_TYPE,
        "metric_type": IndexConstants.METRIC_TYPE,
        "params": {"nlist": IndexConstants.NLIST},
    }
    embedding_collection.create_index(field_name=CollectionConstants.EMBEDDINGS_FIELD_NAME,
                                      index_params=index_parameters)


def generate_embeddings(sentences: List[str]) -> List[float]:
    """It generates the embeddings from the texts

    :param sentences: The input sentences
    :return: the embeddings of the input sentences using an embedding model
    """
    model = EmbeddingModelConstants.MODEL
    return model.encode(sentences)


def insert_new_vectors(sentences, sentences_embeddings, embedding_collection):
    """Inserts the new row in the Collection

    :param sentences: The texts
    :param sentences_embeddings: The embeddings
    :param embedding_collection: The collection
    """
    logger.info("Start inserting entities")
    entities = [
        # We didn't provide the id field because `auto_id` is set to True
        sentences,
        sentences_embeddings,  # field embeddings, supports numpy.ndarray and list
    ]
    embedding_collection.insert(entities)
    embedding_collection.flush()
    logger.info("Number of entities in Milvus: {}".format(embedding_collection.num_entities))


def search_by_embeddings(embedding_collection, input_query, limit):
    """ It loads the collection and conduct a vector search to filter the vectors with the 'sentence' value and displays
    the text_id and the sentence fields of the results.

    :param embedding_collection: The collection
    :param input_query: The input vector query

    """
    embedding_collection.load()
    search_params = {
        "metric_type": SearchConstants.METRIC_TYPE,
        "params": {"nprobe": SearchConstants.NPROBE},
        "offset": SearchConstants.OFFSET
    }

    start_time = time.time()
    results = embedding_collection.search(
        data=[input_query],
        anns_field=CollectionConstants.EMBEDDINGS_FIELD_NAME,
        param=search_params,
        limit=limit,
        expr=None,
        consistency_level="Strong",
        output_fields=[CollectionConstants.TEXT_FIELD_NAME]
    )
    end_time = time.time()
    logger.info("The execution time of the query is {}".format(end_time - start_time))
    result = results[0]
    for i in range(limit):
        logger.info("Result {} is : {}".format(i+1, result[i].entity))
    embedding_collection.release()  # release the memory


def get_texts_and_embeddings():
    """ It returns the sentences and the embeddings given the metadata file.

    :return:
    sentences: The sentences extracted form the file.
    sen_embeddings: Sentences embeddings.
    """

    logger.info("Start extracting sentences from the dataset and generating embeddings")
    df = pd.read_csv(DatasetConstants.METADATA_FILENAME, sep=DatasetConstants.FILE_SEPARATOR,
                     names=[DatasetConstants.WAVES_COLUMN_NAME, DatasetConstants.TEXT_COLUMN_NAME])
    sentences, sen_embeddings = [], []
    for i in tqdm(df.index):
        text = df[DatasetConstants.TEXT_COLUMN_NAME][i]
        sentences.append(text)
        sen_embeddings.append(generate_embeddings(text))
    return sentences, sen_embeddings


def disconnect_from_server():
    """It disconnects from a Milvus server.

    """
    logger.info("disconnecting from Milvus")
    connections.disconnect("default")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sentence", required=True, type=str)
    parser.add_argument("--collection-name", required=True, type=str)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args()
    connect_to_server()

    if utility.has_collection(args.collection_name):
        embedding_collection = Collection(args.collection_name)
    else:
        embedding_collection = create_collection(args.collection_name)
        sentences, sentences_embeddings = get_texts_and_embeddings()
        insert_new_vectors(sentences, sentences_embeddings, embedding_collection)

    # Before conducting a search or a query, you need to load the data in the created collection into memory.
    input_query = generate_embeddings(args.input_sentence)
    search_by_embeddings(embedding_collection, input_query, args.limit)
    disconnect_from_server()


if __name__ == "__main__":
    main()
