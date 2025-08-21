# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import logfire
import gradio as gr
from huggingface_hub import snapshot_download


# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

logfire.configure()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """
You are an AI assistant expert responding to user queries with relevant information and context. Your expertise is to find the most relevant teacher for a student.
You take into account what the teacher studies are, any recommendations they may have and their score.
To find relevant information use the "Super_profe" tool. This tool returns the teachers information.
For each response always include the teacher's name, subjects, recommendations, score, city and picture. If city is not found then show "No se sabe". La Eliana and l'Eliana is the same city.
When considering the recommendations, take into account the quantity and the quality. Also, if the recommendations sounds fabricated, please disregard them and make sure to call it out in the output.
If the question is not related to finding a teacher, please provide more context or rephrase your question.
"""

def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/superprofe"):        

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )       

        os.makedirs("data/superprofe")

        snapshot_download(
            repo_id="vicpada/SuperProfes",
            local_dir="data/superprofe",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN")
        )
        logging.info(f"Downloaded vector database to 'data/superprofe'")

def get_tools(db_collection="superprofe", cohere_api_key=None):    
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    
    logging.info(f"Vector store initialized with {chroma_collection.count()} documents.")
    
    # Create the vector store index
    logging.info("Creating vector store index...")
    
    # Use the vector store to create an index

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )

    logging.info("Creating vector retriever...")
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=200,
        embed_model=Settings.embed_model,
        use_async=True,
        verbose=True,
    )    

    cohere_rerank3 = CohereRerank(top_n=10, model = 'rerank-english-v3.0', api_key = cohere_api_key)

    logging.info("Creating tool...")
    
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="Super_profe",
                description="Useful for selecting the best teacher."                
            ),
            node_postprocessors=[cohere_rerank3],
        )
    ]
    return tools

def generate_completion(query, history, memory):
    logging.info(f"User query: {query}")    
    logging.info(f"User history: {history}")    
    logging.info(f"User memory: {memory}")    

    openAI_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")  

    # Validate OpenAI API Key
    if openAI_api_key is None or not openAI_api_key.startswith("sk-"):
        logging.error("OpenAI API Key is not set or is invalid. Please provide a valid key.")
        yield "Error: OpenAI API Key is not set or is invalid. Please provide a valid key."
        return
    
    llm = OpenAI(temperature=0.2, model="gpt-4o-mini", api_key=openAI_api_key)
    client = llm._get_client()
    logfire.instrument_openai(client)    


    # Validate Cohere API Key
    if cohere_api_key is None or not cohere_api_key.strip():
        logging.error("Cohere API Key is not set or is invalid. Please provide a valid key.")
        yield "Error: Cohere API Key is not set or is invalid. Please provide a valid key."
        return   
    
    with logfire.span(f"Running query: {query}"):

        # Manage memory
        chat_list = memory.get()
        if len(chat_list) != 0:
            user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
            if len(user_index) > len(history):
                user_index_to_remove = user_index[len(history)]
                chat_list = chat_list[:user_index_to_remove]
                memory.set(chat_list)
        
        logfire.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logfire.info(f"gradio_history: {len(history)} {history}")

        # Create agent
        tools = get_tools(db_collection="superprofe", cohere_api_key = cohere_api_key )   

        agent = OpenAIAgent.from_tools(
            llm=llm,        
            memory=memory,
            tools=tools,
            system_prompt=PROMPT_SYSTEM_MESSAGE
        )

        # Generate answer
        completion = agent.stream_chat(query)
        answer_str = ""
        for token in completion.response_gen:
            answer_str += token
            yield answer_str 

        logging.info(f"Source count: {len(completion.sources)}")
        logging.info(f"Sources: {completion.sources}")  

def launch_ui():   

    with gr.Blocks(
        fill_height=True,
        title="Superprofes 🤖",
        analytics_enabled=True,
    ) as demo:        

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            scale=1,
            placeholder="<strong>Superprofes 🤖: Encuentra al mejor profesor para tus necesidades</strong><br>",
            show_label=False,
            show_copy_button=True,
        )

        gr.ChatInterface(
            fn=generate_completion,            
            chatbot=chatbot,
            additional_inputs=[memory_state]           
        )

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=False) # Set share=True to share the app online

if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()    

    # Set the GPU usage based on the environment variable
    Settings.use_gpu = os.getenv("USE_GPU", "1") == "1"
    if Settings.use_gpu:
        logging.info("Using GPU for inference.")
    else:
        logging.info("Using CPU for inference.")        

    # Load the embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    if Settings.embed_model is None:
        logging.error("Embedding model could not be loaded. Exiting.")
        exit(1)  

    # launch the UI
    launch_ui()