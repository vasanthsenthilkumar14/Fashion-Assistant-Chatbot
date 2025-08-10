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

# Generate a response using OpenAI's ChatCompletion API
def generate_response(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message.content

# Chatbot function
def chatbot():
    print("""
    Fashion Assistant: Hello! I can help you find fashion products. 
           What are you looking for?
           (type 'quit' to exit, 'new' to start a new search)
    """)

    # Initial system message for the assistant
    messages = [
        {"role": "system", "content": """You are a helpful assistant in the fashion domain. For each relevant product, include all product details including product ID (p_id) and image URL (img) in your response.
         Guidelines:
            1. For new queries (marked with [NEW]), perform a fresh product search
            2. For follow-up questions (marked with [FOLLOW-UP]), refer to previously mentioned products and context
            3. Always maintain context of the conversation for follow-ups
            4. If clarification is needed, ask follow-up questions
            Note:
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
            3. Any additional recommendations
        """}
    ]

    current_context = None  # Track the current search context

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit condition
        if user_input.lower() == 'quit':
            print("Fashion Assistant: Goodbye!")
            break

        # Check for new search request
        if user_input.lower() == 'new':
            current_context = None
            messages = messages[:1]  # Keep only the system message
            print("Fashion Assistant: Starting a new search. What would you like to find?")
            continue

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
             # Determine if this is a new search or follow-up
            is_new_search = current_context is None
            query_type = "[NEW]" if is_new_search else "[FOLLOW-UP]"

            # Only perform new search if this is a new query
            if is_new_search:
                results = query_collection(user_input)
                results_df = process_results(results)
                reranked_results_df = rerank_results(user_input, results_df)
                top_5_rerank = reranked_results_df.sort_values(by='Reranked_scores', ascending=False).head(5)
                
                # Update current context with search results
                current_context = top_5_rerank
                
                # Format results for inclusion in the response
                formatted_results = []
                for idx, row in top_5_rerank.iterrows():
                    metadata = row['Metadatas']
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
                    formatted_results.append(f"Result {idx+1}: Product Details: {product_details}\nDescription: {row['Documents']}\n")

                result_context = "\n".join(formatted_results)
                assistant_message = f"{query_type}\nBased on the following search results:\n{result_context}\nPlease answer the user query."
            else:
                # For follow-up questions, use existing context
                assistant_message = f"{query_type}\nUsing previous context, please answer: {user_input}"

            # Rest of the conversation handling
            messages.append({"role": "user", "content": assistant_message})
            response = generate_response(messages)
            print("\nFashion Assistant:")
            print(response)
            print("\n")

            # Update conversation history
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": response})

            # Limit conversation history
            if len(messages) > 8:  # 1 system message + 7 recent messages
                messages = messages[:1] + messages[-7:]
            
        except Exception as e:
            print(f"Fashion Assistant: I encountered an error: {str(e)}")
            print("Please try again with a different query.")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
