import tempfile, groq, time, traceback, os
from src.models import *
from src.prompts import *
from src.config import *
from src.loghandler import logger
from src.exceptions import *
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, Form, Depends
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
from llama_index.llms.groq import Groq
from llama_index.core.schema import Document
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core import (
    Settings, 
    SimpleDirectoryReader, 
    # load_index_from_storage, 
    VectorStoreIndex, 
    # StorageContext
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


API_DIR = Path(__file__).resolve().parent / "../"
LOG_FILENAME = str(API_DIR / "./logs/status_logs.log")
DEFAULT_TEMPERATURE = 0.1

class TempAppState:
    chat_memory: ChatMemoryBuffer

class FileUtils:

    ALLOWED_FILES: List = [
        "txt", "csv", "htm", "html", "pdf", "json", "doc", "docx", "pptx"
    ]

    def is_allowed_file(self, filename:str) -> bool:
        return "." in filename and filename.rsplit(".",1)[-1].lower() in self.ALLOWED_FILES

    def run_file_checks(self, files: List[UploadFile]):

        if not files:
            message = f"No file found"
            logger.error(message)
            return JSONResponse(
                content={
                    "status": message},
                status_code=400
            )
        
        for file in files:
            filename = file.filename
            if not file or filename == "":
                message = f"No selected file"
                logger.error(message)
                return JSONResponse(
                    content ={
                        "status": message
                    },
                    status_code=400
                )
            
            if not self.is_allowed_file(filename):
                message = f"File format {filename.rsplit('.',1)[-1].lower()} not supported. Use any of {self.ALLOWED_FILES}"
                logger.warning(message)
                return JSONResponse(
                    content={"status": message},
                    status_code=415
                )
        
        return JSONResponse(
            content={"status": "success"},
            status_code=200
        )

    async def upload_files(
        self,
        files: List[UploadFile], 
        temp_dir: tempfile.TemporaryDirectory
    ):
        file_checks = self.run_file_checks(files)
        if file_checks.status_code==200:
            filename = ""
            try:
                for file in files:
                    filename = file.filename
                    filepath = os.path.join(temp_dir, filename)
                    file_obj = await file.read()

                    with open(filepath, "wb") as buffer:
                        buffer.write(file_obj)
        
                message = f"Files uploaded successfully."
                logger.info(message)
                return JSONResponse(
                    content={"status": message},
                    status_code=200
                )
            
            except Exception as e:
                message = f"An error occured while trying to upload the file, {filename}: {e}"
                logging.error(message)
                raise UploadError(message)
            
        raise FileCheckError(file_checks["status"])

class EmbeddingUtils:

    tokenizer = SentenceSplitter()._tokenizer
    text_splitter = SentenceSplitter()._split_text

    embed_model: str = DEFAULT_EMBED_MODEL
    embed_func = HuggingFaceEmbedding(
        model_name=embed_model,
    )

    async def generate_and_store_embeddings(
        self,
        chat_uid: str,
        documents: List[Document]
    ):

        collection_name = f"aisoc-{chat_uid}-embeddings"
        chroma_collection = ChromaUtils().init_chroma(collection_name, task="create")

        try:
            logger.info(f"Generating vector embeddings for collection: {collection_name}...")
            start_time = time.time()

            doc_split_by_chunk_size = [
                (
                    [
                        {"content": item, "metadata": doc.metadata}
                        for item in self.text_splitter(doc.text, chunk_size=1024)
                    ]
                    if len(self.tokenizer(doc.text)) > 1536
                    else [{"content": doc.text, "metadata": doc.metadata}]
                )
                for doc in documents
            ]  # nested list

            doc_chunks = sum(doc_split_by_chunk_size, []) # flatten nested list
            content_list = [doc["content"] for doc in doc_chunks]
            metadata_list = [doc["metadata"] for doc in doc_chunks]

            id_list = [f"embedding-{i+1}" for i in range(len(content_list))]

            embeddings = [self.embed_func.get_text_embedding(item) for item in content_list]
            # logger.info(f"Document token sizes: {[len(self.tokenizer(item)) for item in content_list]}")
            logger.info(f"Embeddings generated for collection: {collection_name} in {time.time()-start_time} seconds.")

        except Exception as e:
            message = f"Error generating embeddings for collection: {collection_name}. Error: {e}"
            logger.error(message)
            raise EmbeddingError(message)

        # populate chroma collection with embeddings
        logger.info(f"Populating collection {collection_name} with computed embeddings...")
        chroma_collection.upsert(
            ids=id_list,
            documents=content_list,
            metadatas=metadata_list,
            embeddings=embeddings
        )

        # inspect collection
        collection_count = chroma_collection.count()
        if collection_count == 0:
            message = f"Could not store embeddings in Chroma database. Collection is empty!"
            logger.error(message)
            raise ChromaCollectionError(message)

        logger.info(f"Collection size::{collection_count}")

    async def retrieve_embeddings(self, chat_uid: str):

        collection_name = f"aisoc-{chat_uid}-embeddings"
        chroma_collection = ChromaUtils().init_chroma(collection_name)

        collection_count = chroma_collection.count()
        if collection_count == 0:
            message = f"Could not find embeddings in ChromaDB for conversation {chat_uid}. Please pass the correct chat_uid."
            logger.error(message)
            raise ChromaCollectionError(message)

        chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embeddings = VectorStoreIndex.from_vector_store(
            vector_store=chroma_vector_store,
            embed_model=self.embed_func
        )
        logger.info(f"Embeddings retrieved from ChromaDB for collection {collection_name}")

        return embeddings, collection_count

class ChromaUtils:

    def get_chroma_client(self, use_server: bool = True):
        
        """
        Initialize Chroma in server mode by default and provide CHROMA_SERVER_HOST/PORT
        If you do not want to use an external server, set CHROMA_USE_SERVER=false; this will use ChromaDB persistent client mode
        """

        if use_server:
            # Only use server mode if explicitly requested
            if CHROMADB_SSL:
                chroma_client = chromadb.HttpClient(
                    host=CHROMADB_HOST, port=CHROMADB_PORT, ssl=True
                )
            else:
                chroma_client = chromadb.HttpClient(
                    host=CHROMADB_HOST, port=CHROMADB_PORT
                )
        else:
            # Embedded, on-disk Chroma (recommended for local dev)
            chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
            # For recent Chroma, 'path' is the kw. Older versions may use 'persist_directory'.
            try:
                chroma_client = chromadb.PersistentClient(path=chroma_path)
            except TypeError:
                chroma_client = chromadb.PersistentClient(persist_directory=chroma_path)

        return chroma_client

    def init_chroma(self, collection_name: str, task: str = "retrieve"):
        # use_server = os.getenv("CHROMA_USE_SERVER", "false").lower() in ("1", "true", "yes")
        logger.info(f"Initializing Chroma database...")
        try:
            chroma_client = self.get_chroma_client(use_server=CHROMA_USE_SERVER)
            
        except Exception as e:
            message = f"Error connecting to Chroma database for collection `{collection_name}`: {e}"
            logger.error(message)
            raise ChromaConnectionError(message)

        collection = chroma_client.get_or_create_collection(
            collection_name,
            embedding_function=None, # we are not passing an embedding_func here as we are handling embedding generation in generate_and_store_embeddings()
            metadata={"hnsw": "cosine"}
        )
        logger.info(f"Collection {task}d: {collection_name}")
        return collection

class ChatHistoryManager:
    """
    Manages chat history storage and retrieval using ChromaDB.
    Stores chat messages using the same embedding function as documents.
    """
    
    def __init__(self):
        self.chroma_utils = ChromaUtils()
        # Use the same embedding function as document embeddings for consistency
        self.embed_func = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
    
    def _init_chat_history_collection(self, collection_name: str):
        """Initialize ChromaDB collection specifically for chat history"""
        logger.info(f"Initializing Chroma database for chat history...")
        try:
            chroma_client = self.chroma_utils.get_chroma_client(use_server=CHROMA_USE_SERVER)
            
        except Exception as e:
            message = f"Error connecting to Chroma database for collection `{collection_name}`: {e}"
            logger.error(message)
            raise ChromaConnectionError(message)

        # Create collection without embedding function - we'll provide embeddings manually
        collection = chroma_client.get_or_create_collection(
            collection_name,
            embedding_function=None,
            metadata={"hnsw": "cosine"}
        )
        logger.info(f"Chat history collection initialized: {collection_name}")
        return collection
    
    def upsert_message_pair(
        self,
        chat_uid: str,
        user_query: str,
        assistant_response: str,
        conversation_turn: int,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Store both user query and assistant response as a complete conversation turn.
        
        Args:
            chat_uid: Unique identifier for the chat session
            user_query: The user's message
            assistant_response: The assistant's response
            conversation_turn: Sequential turn number for this pair
            timestamp: Optional timestamp for both messages
            
        Returns:
            bool: True if both messages stored successfully
        """
        
        collection_name = f"aisoc-{chat_uid}-chat-history"
        
        try:
            # Get or create the chat history collection
            chroma_collection = self._init_chat_history_collection(collection_name)
            
            # Generate timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            # Create unique IDs for both messages
            user_id = f"{chat_uid}-turn-{conversation_turn:04d}-user"
            assistant_id = f"{chat_uid}-turn-{conversation_turn:04d}-assistant"
            
            # Generate embeddings for both messages
            user_embedding = self.embed_func.get_text_embedding(user_query)
            assistant_embedding = self.embed_func.get_text_embedding(assistant_response)
            
            # Prepare metadata for both messages
            user_metadata = {
                "chat_uid": chat_uid,
                "message_type": "user",
                "conversation_turn": conversation_turn,
                "timestamp": timestamp,
                "message_id": user_id
            }
            
            assistant_metadata = {
                "chat_uid": chat_uid,
                "message_type": "assistant", 
                "conversation_turn": conversation_turn,
                "timestamp": timestamp,
                "message_id": assistant_id
            }
            
            # Upsert both messages with their embeddings
            chroma_collection.upsert(
                ids=[user_id, assistant_id],
                documents=[user_query, assistant_response],
                metadatas=[user_metadata, assistant_metadata],
                embeddings=[user_embedding, assistant_embedding]
            )
            
            logger.info(f"Message pair stored in collection {collection_name}: turn {conversation_turn}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chat message pair: {e}")
            return False
    
    def retrieve_chat_history(
        self,
        chat_uid: str,
        as_pairs: bool = True,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve chat history from ChromaDB.
        Returns a list of ALL message pairs in the chat history.
        
        Args:
            chat_uid: Unique identifier for the chat session
            as_pairs: If True, returns list of {user: str, assistant: str} dicts
                     If False, returns list of individual messages with metadata
            limit: Optional limit on number of pairs/messages to return
            
        Returns:
            List[Dict]: Complete chat history as message pairs or individual messages
        """
        
        collection_name = f"aisoc-{chat_uid}-chat-history"
        
        try:
            # Get the chat history collection
            chroma_collection = self._init_chat_history_collection(collection_name)
            
            # Check if collection exists and has data
            collection_count = chroma_collection.count()
            if collection_count == 0:
                logger.warning(f"No chat history found for conversation {chat_uid}")
                return []
            
            # Retrieve ALL messages from the collection
            results = chroma_collection.get(
                include=["documents", "metadatas"]
            )
            
            # Combine documents with their metadata
            messages = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                messages.append({
                    "content": doc,
                    "message_type": metadata["message_type"],
                    "conversation_turn": metadata["conversation_turn"],
                    "timestamp": metadata["timestamp"],
                    "message_id": metadata["message_id"]
                })
            
            # Sort by conversation turn to maintain chronological order
            messages.sort(key=lambda x: x["conversation_turn"])
            
            if as_pairs:
                # Group messages into pairs - THIS RETURNS ALL MESSAGE PAIRS
                message_pairs = []
                current_pair = {}
                
                for message in messages:
                    turn = message["conversation_turn"]
                    msg_type = message["message_type"]
                    content = message["content"]
                    
                    if msg_type == "user":
                        # Start a new pair
                        current_pair = {
                            "turn": turn,
                            "user": content,
                            "assistant": None,
                            "timestamp": message["timestamp"]
                        }
                    elif msg_type == "assistant" and current_pair:
                        # Complete the current pair
                        current_pair["assistant"] = content
                        message_pairs.append(current_pair)
                        current_pair = {}
                
                # Apply limit if specified (but by default returns ALL pairs)
                if limit:
                    message_pairs = message_pairs[-limit:]
                
                logger.info(f"Retrieved {len(message_pairs)} message pairs from {collection_name}")
                return message_pairs  # RETURNS LIST OF ALL MESSAGE PAIRS
            
            else:
                # Return individual messages
                if limit:
                    messages = messages[-limit*2:]
                
                logger.info(f"Retrieved {len(messages)} individual messages from {collection_name}")
                return messages
                
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            raise ChromaCollectionError(f"Could not retrieve chat history for {chat_uid}: {e}")

class ChatEngine:
    
    async def generate_response(
        self,
        query: str,
        chat_uid: str,
        model: str = LLAMA_3_3_70B,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chatbot_name: str = "",
        chat_mode: str = "context",
        verbose: bool = True,
        streaming: bool = True,
        app_state: TempAppState = None
    ):
        chatbot_desc = f"Your name is {chatbot_name}. " if chatbot_name else ""
        system_prompt = system_prompt.format(chatbot_desc=chatbot_desc)
        logger.info(f"System prompt::{system_prompt}")

        index, index_size = await EmbeddingUtils().retrieve_embeddings(chat_uid)
        Settings.llm = LLMClient().map_task_to_client(task="rag", model=model)
        # Settings.embed_model = HuggingFaceEmbedding()

        # heuristic for choice_k; experiment until you achieve optimal rule
        # don't use groq models for longer contexts due to rate limit on free plan
        choice_k = 3 if index_size < 20 \
            else 5 if index_size < 50 \
                else index_size//10 if index_size < 200 \
                    else 30
        
        app_state.chat_memory = self.retrieve_chat_memory(choice_k=choice_k, app_state=app_state)

        chat_engine = index.as_chat_engine(
            # llm=llm_client,
            chat_mode=chat_mode,
            system_prompt=system_prompt,
            similarity_top_k=choice_k,
            verbose=verbose,
            streaming=streaming,
            memory=app_state.chat_memory
        )

        response = chat_engine.stream_chat(query)
        logger.info("Starting response stream...\n")
        try:
            for token in response.response_gen:
                print(token, end="")
                yield str(token)
        except:
            message = f"An error occured while generating chat response."
            exception = traceback.format_exc()
            logger.error(f"{message}: {exception}")
            raise ChatEngineError(f"{message}. See the system logs for more information.")

    # Methods for managing chat history within an API session - not ideal for production
    def init_chat_memory(self, choice_k):
        token_limit = choice_k*1024
        return ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    def retrieve_chat_memory(self, choice_k, app_state:TempAppState=None):
        try:
            logger.info("Retrieving chat memory...")
            return app_state.chat_memory or self.init_chat_memory(choice_k)
        except:
            logger.warning("Could not retrieve chat memory. Creating new memory...")
            return self.init_chat_memory(choice_k)