# milvus-esperanto

### Requirements 
* Python version >= 3.8
* Milvus version >= 2.2

#### To install Milvus Standalone with Docker Compose : 
1- Download the YAML file:
```commandline
$ wget https://github.com/milvus-io/milvus/releases/download/v2.2.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
2- Start Milvus by running the following command in the same directory as the docker-compose.yml : 
* If your system has Docker Compose V1 : 
```commandline
$ sudo docker-compose up -d
```
* If your system has Docker Compose V2 : 
```commandline
$ sudo docker compose up -d
```
#### For more installation options please refer to [Milvus documentation](https://milvus.io/docs/install_standalone-helm.md).

### To run the script text_embeddings.py:
1- Install the requirements by running :
```commandline
pip install -r requirements.txt
```
2- Run the script : 
```commandline
python3 text_embeddings.py --input-sentence "The_input_text" --collection-name "the_name_of_the_collection" --limit the_number_of_the_most_similar_results_to_return
```
##### NOTE : This script allows you to create a new collection if it does not exist and to conduct a text similarity search.