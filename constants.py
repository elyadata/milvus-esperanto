from sentence_transformers import SentenceTransformer


class EmbeddingModelConstants:
    DIM = 768  # The dimension of the embedding vector
    MODEL = SentenceTransformer('sentence-transformers/LaBSE')


class DatasetConstants:
    TEXT_COLUMN_NAME = "text"
    WAVES_COLUMN_NAME = "wav"
    METADATA_FILENAME = "orthographic-transcript.txt"
    FILE_SEPARATOR = ' '


class MilvusServerConstants:
    HOST = "localhost"
    PORT = "19530"


class CollectionConstants:
    ID_FIELD_NAME = "text_id"
    TEXT_FIELD_NAME = "sentence"
    EMBEDDINGS_FIELD_NAME = "embeddings"


class IndexConstants:
    INDEX_TYPE = "IVF_SQ8"
    METRIC_TYPE = "L2"
    NLIST = 128  # This is the number of cluster units


class SearchConstants:
    METRIC_TYPE = "L2"
    NPROBE = 10
    OFFSET = 8
