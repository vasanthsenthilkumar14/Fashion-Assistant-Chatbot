import pandas as pd
import yaml
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sentence_transformers import CrossEncoder, util
import openai

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


# OpenAI Moderation Function
def check_moderation(text):
    """
    Checks if the input text violates OpenAI's moderation guidelines.
    
    Args:
        text (str): The text to check for moderation violations.
    
    Returns:
        dict: Contains a flag for any violations and the categories flagged.
    """
    try:
        response = openai.moderations.create(input=text)
        results = response.results[0]
        
        return {
            "flagged": results.flagged,
            "categories": {
                cat: flagged 
                for cat, flagged in results.categories.model_dump().items() 
                if flagged
            }
        }
    except Exception as e:
        print(f"Error during moderation check: {e}")
        return {"flagged": False, "categories": {}}


def query_collection(query):
    fashion_collection = client.get_collection(name='RAG_on_Fashion', embedding_function=embedding_function)
    results = fashion_collection.query(
        query_texts=query,
        n_results=10
    )
    return results

def process_results(results):
    result_dict = {
        'Metadatas': results['metadatas'][0],
        'Documents': results['documents'][0],
        'Distances': results['distances'][0],
        'IDs': results['ids'][0]
    }
    results_df = pd.DataFrame.from_dict(result_dict)
    return results_df

def rerank_results(query, results_df):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    results_df['Reranked_scores'] = cross_rerank_scores
    return results_df

# Define the function to generate the response. Provide a comprehensive prompt that passes the user query and the top 3 results to the model
def generate_response(query, top_5_RAG):
    """
    Generate a response using GPT-4o's ChatCompletion based on the user query and retrieved information.
    """
    # Format the search results to be more readable and include all metadata
    formatted_results = []
    for idx, row in top_5_RAG.iterrows():
        doc = row['Documents']
        metadata = row['Metadatas']  # Remove eval() since metadata is already a dict
        product_details = {
            'name': metadata.get('name', ''),
            'brand': metadata.get('Brand', ''),
            'price': metadata.get('Price', ''),
            'color': metadata.get('Color', ''),
            'rating': metadata.get('Rating', ''),
            'rating_count': metadata.get('Rating_Count', ''),
            'p_id': metadata.get('p_id', ''),
            'img': metadata.get('img', ''),
            'attributes': metadata.get('Attributes', {})
        }
        formatted_results.append(f"Result {idx+1}:\nProduct Details: {product_details}\nDescription: {doc}\n")
    
    formatted_results_str = "\n".join(formatted_results)

    messages = [
        {"role": "system", "content": "You are a helpful assistant in the fashion domain. For each relevant product, include all product details including product ID (p_id) and image URL (img) in your response."},
        {"role": "user", "content": f"""Based on the following search results:
{formatted_results_str}
Please answer the user query: {query}

Guidelines:
1. Provide relevant and accurate suggestions (all matching products) based on the query and available data
2. For EACH relevant product suggestion, include:
   - Product name
   - Brand
   - Price
   - Color
   - Rating and review count
   - Product ID (p_id)
   - Image URL (img)
   - Relevant attributes
3. Format each product suggestion in a clear, structured way
4. If query is irrelevant to the data, state that clearly

Please structure your response with:
1. Answer to the query with all matching suggestions
2. Detailed product suggestions with all metadata
3. Any additional recommendations"""}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )
    
    return response.choices[0].message.content.split('\n')

def chatbot():
    print("""
    Fashion Assistant: Hello! I can help you find fashion products. 
           What are you looking for?
           (type 'quit' to exit)
    """)
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit condition
        if user_input.lower() == 'quit':
            print("Fashion Assistant: Goodbye!")
            break
            
        # Skip empty inputs
        if not user_input:
            continue
            
        # Moderation check for user input
        moderation_result = check_moderation(user_input.lower())
        if moderation_result["flagged"]:
            print("Fashion Assistant: Your input contains inappropriate content. Please try again.")
            print(f"Flagged categories: {', '.join(moderation_result['categories'].keys())}")
            continue
        
        try:
            # Process the query
            results = query_collection(user_input)
            results_df = process_results(results)
            
            # Rerank results
            reranked_results_df = rerank_results(user_input, results_df)
            top_5_rerank = reranked_results_df.sort_values(by='Reranked_scores', ascending=False).head(5)
            top_5_RAG = top_5_rerank[["Documents", "Metadatas"]].head(5)
            
            # Generate and display response
            response = generate_response(user_input, top_5_RAG)
            print("\nFashion Assistant:")
            for line in response:
                print(line)
            print("\n")
            
        except Exception as e:
            print(f"Fashion Assistant: I encountered an error: {str(e)}")
            print("Please try again with a different query.")

# Update the main block to use the chatbot
if __name__ == "__main__":
    chatbot()