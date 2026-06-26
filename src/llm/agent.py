"""
LLM Agent with RAG Tool Integration

This defines the customer support agent that uses a Language Model
with Retrieval-Augmented Generation (RAG) capabilities.

"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


class BaseAgent(ABC):
    """
    Abstract base class for LLM agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            config: Configuration dictionary containing LLM settings, prompts, etc.
        """
        self.config = config or {}
        self.is_initialized = False
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent with LLM and tools."""
        pass
    
    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process a text query and return a response.
        
        Args:
            text: Input text from the user
            **kwargs: Additional context or parameters
            
        Returns:
            str: Agent's response
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent implementation using LangChain ReAct agent.
    
    This agent uses a Language Model with RAG capabilities to answer
    customer support queries by retrieving relevant information from
    a knowledge base.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.knowledge_base = None
        
    async def initialize(self) -> None:
        try:
            from langchain_openai import ChatOpenAI
            import os

            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = self.config.get("model", "gpt-3.5-turbo")
            temperature = self.config.get("temperature", 0.7)
            base_url = self.config.get("base_url")

            llm_kwargs = {
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
            }
            if base_url:
                llm_kwargs["base_url"] = base_url

            self.llm = ChatOpenAI(**llm_kwargs)

            await self._setup_knowledge_base()
            tools = await self._create_tools()
            await self._create_agent(tools)

            self.is_initialized = True
            print(f"CustomerSupportAgent initialized successfully (model={model})")

        except Exception as e:
            print(f"Agent initialization error: {str(e)}")
            raise
    
    async def _setup_knowledge_base(self) -> None:
        """
        Set up the knowledge base for RAG using ChromaDB.
        
        This method automatically creates embeddings and stores them in ChromaDB.
        Students only need to implement the retrieval logic in _rag_search().
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import os
            import hashlib
            
            # Initialize ChromaDB (persistent storage)
            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Collection name
            collection_name = "customer_support_kb"
            
            # Check if collection already exists and has data
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    print(f"Knowledge base already exists with {self.collection.count()} documents")
                    return
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"}
                )
            
            # Load predefined customer support documents
            knowledge_documents = self._get_customer_support_documents()
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Process and store documents
            print(f"Ingesting {len(knowledge_documents)} documents into knowledge base...")
            
            documents = []
            metadatas = []
            ids = []
            
            for i, doc_data in enumerate(knowledge_documents):
                doc_id = f"doc_{i}_{hashlib.md5(doc_data['content'].encode()).hexdigest()[:8]}"
                
                documents.append(doc_data['content'])
                metadatas.append({
                    'category': doc_data['category'],
                    'title': doc_data['title'],
                    'doc_id': doc_id
                })
                ids.append(doc_id)
            
            # Add documents to ChromaDB (it will automatically create embeddings)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully ingested {len(documents)} documents into ChromaDB")
            
        except Exception as e:
            print(f"Error setting up knowledge base: {str(e)}")
            raise
    
    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """
        Predefined customer support knowledge base.
        
        This is the definitive knowledge base that students will work with.
        Do not modify these documents - they form the complete knowledge base.
        """
        return [
            # Return Policy
            {
                "title": "Return Policy Overview",
                "category": "returns",
                "content": "We offer a 30-day return policy for all products purchased from our store. Items must be in original condition with all tags and packaging intact. Returns are processed within 5-7 business days of receiving the returned item. Refunds are issued to the original payment method."
            },
            {
                "title": "Return Process Steps",
                "category": "returns", 
                "content": "To initiate a return: 1) Log into your account and go to Order History, 2) Select the order and click 'Return Items', 3) Choose the items to return and reason, 4) Print the prepaid return label, 5) Pack items securely and attach the label, 6) Drop off at any UPS location or schedule pickup."
            },
            {
                "title": "Non-Returnable Items",
                "category": "returns",
                "content": "The following items cannot be returned: personalized or customized products, perishable goods, digital downloads, gift cards, intimate apparel, and items marked as final sale. Health and safety regulations prevent returns of opened cosmetics and personal care items."
            },
            
            # Shipping Information
            {
                "title": "Shipping Methods and Times",
                "category": "shipping",
                "content": "We offer multiple shipping options: Standard shipping (5-7 business days, free on orders over $50), Express shipping (2-3 business days, $12.99), Next-day shipping (1 business day, $24.99). All orders placed before 2 PM EST ship the same day."
            },
            {
                "title": "International Shipping",
                "category": "shipping",
                "content": "We ship internationally to over 50 countries. International shipping takes 7-14 business days via DHL Express. Shipping costs vary by destination and are calculated at checkout. Customers are responsible for customs fees and import duties. Some restrictions apply to certain products and countries."
            },
            {
                "title": "Order Tracking",
                "category": "shipping",
                "content": "Once your order ships, you'll receive a tracking number via email. Track your package using the tracking number on our website or the carrier's website. You can also track orders by logging into your account and viewing Order History. Tracking updates may take 24 hours to appear."
            },
            
            # Customer Support
            {
                "title": "Contact Information",
                "category": "support",
                "content": "Customer support is available 24/7 via multiple channels: Phone: 1-800-HELP-NOW (1-800-435-7669), Email: support@company.com, Live chat on our website (available 6 AM - 12 AM EST), or submit a support ticket through your account dashboard."
            },
            {
                "title": "Response Times",
                "category": "support",
                "content": "Our support team response times: Live chat - immediate during business hours, Phone support - average wait time under 3 minutes, Email support - response within 4 hours during business days, Support tickets - response within 24 hours. Premium customers receive priority support with faster response times."
            },
            
            # Warranty and Technical Support
            {
                "title": "Product Warranty",
                "category": "warranty",
                "content": "All products come with a manufacturer's warranty. Electronics have 1-year warranty covering defects and malfunctions. Apparel and accessories have 90-day warranty against material defects. Warranty claims require proof of purchase and must be initiated within the warranty period."
            },
            {
                "title": "Technical Support",
                "category": "technical",
                "content": "Free technical support is available for all electronic products. Our certified technicians provide assistance with setup, troubleshooting, and software issues. Technical support is available Monday-Friday 8 AM - 8 PM EST via phone or email. We also offer remote assistance for compatible devices."
            },
            
            # Account and Orders
            {
                "title": "Account Management",
                "category": "account",
                "content": "Manage your account online: Update personal information and addresses, view order history and tracking, manage payment methods, set communication preferences, download invoices and receipts. Account changes may take up to 24 hours to reflect across all systems."
            },
            {
                "title": "Order Modifications",
                "category": "orders",
                "content": "Orders can be modified or canceled within 1 hour of placement if not yet processed. Contact customer service immediately to make changes. Once an order is processed and shipped, it cannot be modified. You can return unwanted items following our return policy."
            },
            
            # Payment and Billing
            {
                "title": "Payment Methods",
                "category": "payment",
                "content": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later options through Klarna and Afterpay. Gift cards and store credit can also be used for purchases. Payment is processed securely using 256-bit SSL encryption."
            },
            {
                "title": "Billing and Invoices",
                "category": "billing",
                "content": "Billing occurs when your order ships. You'll receive an email confirmation with invoice details. Invoices are available in your account under Order History. For business purchases, we can provide detailed invoices with tax information. Contact our billing department for any payment disputes or questions."
            },
            
            # Product Information
            {
                "title": "Product Availability",
                "category": "products",
                "content": "Product availability is updated in real-time on our website. If an item shows as 'In Stock', it's available for immediate shipping. 'Limited Stock' means fewer than 10 items remaining. 'Pre-order' items will ship on the specified release date. Out of stock items can be added to your wishlist for restock notifications."
            },
            {
                "title": "Size and Fit Guide",
                "category": "products",
                "content": "Each product page includes detailed size charts and fit information. For apparel, we recommend checking measurements against our size guide rather than relying on size labels from other brands. If you're between sizes, we generally recommend sizing up. Our customer service team can provide personalized fit recommendations."
            }
        ]
    
    async def _create_tools(self) -> List[Tool]:
        """
        TODO: Create tools for the agent, including the RAG tool.
        
        Returns:
            List[Tool]: List of tools available to the agent
        """
        tools = []
        
        # TODO: Implement RAG tool
        rag_tool = Tool(
            name="knowledge_search",
            description="Search the customer support knowledge base for relevant information",
            func=self._rag_search
        )
        tools.append(rag_tool)
        
        # TODO: Add more tools as needed (e.g., order lookup, shipping tracker)
        
        return tools
    
    def _rag_search(self, query: str) -> str:
        """
        TODO: Implement embedding-based retrieval from ChromaDB.
        
        The knowledge base is already set up with documents and embeddings.
        You need to implement the search logic to retrieve relevant information.
        
        Args:
            query: Search query from the user (e.g., "What is your return policy?")
            
        Returns:
            str: Formatted relevant information from the knowledge base
            
        Available resources:
        - self.collection: ChromaDB collection with embedded documents
        - self.embedding_model: SentenceTransformer model for creating query embeddings
        
        Implementation steps:
        1. Use ChromaDB's query method to search for relevant documents
        2. Retrieve top 3-5 most relevant documents
        3. Format the results into a readable response
        4. Include document titles and categories for context
        
        ChromaDB query example:
        ```python
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        ```
        
        The results structure:
        - results['documents'][0]: List of document contents
        - results['metadatas'][0]: List of metadata (title, category)
        - results['distances'][0]: List of similarity distances (lower = more similar)
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return "Knowledge base not available. Please ensure the service is properly initialized."

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )

            if not results['documents'] or not results['documents'][0]:
                return "No relevant information found in the knowledge base for your query."

            formatted_results = []
            for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                relevance = round((1 - distance) * 100, 1)
                formatted_results.append(
                    f"**{meta.get('title', 'Document')}** (Category: {meta.get('category', 'general')}, Relevance: {relevance}%)\n{doc}"
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    async def _create_agent(self, tools: List[Tool]) -> None:
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.agents import create_react_agent, AgentExecutor
        from langchain_classic.memory import ConversationBufferMemory

        prompt_template = """You are a professional and helpful customer support agent. Your job is to assist customers with their queries about orders, returns, shipping, payments, warranties, and account management.

Always be polite, clear, and accurate. Use the knowledge_search tool to find relevant information before answering.

You have access to the following tools:
{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: think about what information you need
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (you can repeat Thought/Action/Action Input/Observation multiple times if needed)
Thought: I now know the final answer
Final Answer: provide a helpful, complete answer to the customer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(prompt_template)

        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def process_query(self, text: str, **kwargs) -> str:
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        if not text or not text.strip():
            return "I didn't receive any text to process. Could you please repeat your question?"

        try:
            result = await self.agent_executor.ainvoke({"input": text.strip()})
            return result.get("output", "I'm sorry, I could not process your request.")
        except Exception as e:
            print(f"Agent processing error: {str(e)}")
            try:
                rag_result = self._rag_search(text)
                if rag_result and "TODO" not in rag_result:
                    return f"Based on our knowledge base: {rag_result}"
            except Exception:
                pass
            return "I apologize, but I'm having trouble processing your request right now. Please contact our support team directly."

    async def cleanup(self) -> None:
        try:
            self.llm = None
            self.agent = None
            self.agent_executor = None
            self.is_initialized = False
            print("Agent cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
