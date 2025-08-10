import pandas as pd
import yaml
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np

# Load API keys from config
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return {}
    
    # Load configuration data
config_data = load_yaml('./config.yaml')
os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]

# Set up PersistentClient with the path for saving data
chroma_data_path = './ChromaDB_Data_v2'
client = chromadb.PersistentClient(path=chroma_data_path)

# Set up the embedding function using OpenAI embedding model
model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name=model)

def chunk_data(main_csv_path, chunk_size=1000):
    # Load dataset
    main_df = pd.read_csv(main_csv_path)
    main_df = main_df.dropna()
    
    # Calculate number of chunks needed
    n_chunks = len(main_df) // chunk_size + (1 if len(main_df) % chunk_size != 0 else 0)
    chunks = np.array_split(main_df, n_chunks)
    return chunks

def prepare_data(chunk):
    fashion_df = chunk.copy()
    # Combine relevant columns for embedding and metadata
    fashion_df["Combined"] = (
        "Product: " + fashion_df["name"].str.strip() +
        "; Brand: " + fashion_df["brand"].str.strip() +
        "; Color: " + fashion_df["colour"].str.strip() +
        "; Price: " + fashion_df["price"].astype(str) +
        "; Rating: " + fashion_df["avg_rating"].astype(str) +
        " (" + fashion_df["ratingCount"].astype(str) + " reviews)" +
        "; Attributes: " + fashion_df["p_attributes"].str.strip()
    )

    fashion_df['Metadata'] = fashion_df.apply(lambda x: {
        'p_id': x['p_id'],
        'name': x['name'],
        'Product_Name': x['products'],
        'Brand': x['brand'],
        'Color': x['colour'],
        'Price': x['price'],
        'Rating': x['avg_rating'],
        'Rating_Count': x['ratingCount'],
        'Attributes': x['p_attributes'],
        'Description': x['description'],
        'img': x['img']
    }, axis=1)

    return fashion_df

def embedding_and_storing_data(fashion_df):
    # Initialize the collection
    fashion_collection = client.get_or_create_collection(name='RAG_on_Fashion', embedding_function=embedding_function)

    # Prepare documents and metadata for insertion
    documents_list = fashion_df["Combined"].tolist()
    metadata_list = fashion_df['Metadata'].tolist()
    print("Number of documents:", len(documents_list))

    fashion_collection.add(
        documents=documents_list,
        ids=fashion_df['p_id'].astype(str).tolist(),
        metadatas=metadata_list
    )

    print("Number of documents in ChromaDB: ", fashion_collection.count())
    print("Data added to ChromaDB successfully.")
    return 1

if __name__ == "__main__":
    # Load dataset and split into chunks
    file_path = 'Fashion Dataset v2.csv'
    chunks = chunk_data(file_path)
    print("Total number of chunks: ", len(chunks))

    for chunk_index, chunk in enumerate(chunks):
        print("Processing chunk: ", chunk_index)
        fashion_df = prepare_data(chunk)
        embedding_and_storing_data(fashion_df)
