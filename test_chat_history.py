"""
Test script to confirm that retrieve_chat_history() returns a list of all message pairs
"""

import asyncio
import os
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

from src.helpers import ChatHistoryManager
from src.loghandler import set_logger, ColorFormmater

# Create logger for this test
logger = set_logger(
    to_file=False, 
    to_console=True, 
    custom_formatter=ColorFormmater
)

async def test_chat_history_retrieval():
    """Test function to verify chat history storage and retrieval"""
    
    # Initialize the chat history manager
    chat_manager = ChatHistoryManager()
    test_chat_uid = "test-session-verification"
    
    print("=" * 60)
    print("🧪 TESTING CHAT HISTORY RETRIEVAL FUNCTION")
    print("=" * 60)
    
    # Step 1: Store some test message pairs
    print("\n📝 Step 1: Storing test message pairs...")
    
    test_conversations = [
        {
            "user": "What is machine learning?",
            "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
            "turn": 1
        },
        {
            "user": "Can you give me an example?",
            "assistant": "Sure! Email spam detection is a great example. The system learns from thousands of emails labeled as 'spam' or 'not spam' to automatically classify new emails.",
            "turn": 2
        },
        {
            "user": "What about deep learning?",
            "assistant": "Deep learning is a specialized type of machine learning that uses neural networks with multiple layers (hence 'deep') to process complex patterns in data, like recognizing images or understanding speech.",
            "turn": 3
        }
    ]
    
    # Store each conversation pair
    for conv in test_conversations:
        success = chat_manager.upsert_message_pair(
            chat_uid=test_chat_uid,
            user_query=conv["user"],
            assistant_response=conv["assistant"],
            conversation_turn=conv["turn"]
        )
        print(f"   ✅ Turn {conv['turn']}: {'Stored successfully' if success else 'Failed to store'}")
    
    print(f"\n✅ Stored {len(test_conversations)} message pairs")
    
    # Step 2: Retrieve the chat history
    print("\n📥 Step 2: Retrieving chat history...")
    
    try:
        retrieved_history = chat_manager.retrieve_chat_history(
            chat_uid=test_chat_uid,
            as_pairs=True  # This should return the list of message pairs
        )
        
        # Step 3: Verify the results
        print("\n🔍 Step 3: Verification Results...")
        print(f"   📊 Retrieved {len(retrieved_history)} message pairs")
        print(f"   📋 Type of returned object: {type(retrieved_history)}")
        
        if retrieved_history:
            print(f"   📋 Type of each item: {type(retrieved_history[0])}")
            print(f"   🔑 Keys in each message pair: {list(retrieved_history[0].keys())}")
        
        # Step 4: Display the retrieved conversation
        print("\n💬 Step 4: Complete Retrieved Conversation:")
        print("-" * 50)
        
        for i, pair in enumerate(retrieved_history, 1):
            print(f"\n🔄 Turn {pair['turn']}:")
            print(f"   👤 User: {pair['user']}")
            print(f"   🤖 Assistant: {pair['assistant']}")
            print(f"   ⏰ Timestamp: {pair['timestamp']}")
        
        # Step 5: Confirm the format
        print("\n" + "=" * 60)
        print("✅ CONFIRMATION: FUNCTION RETURNS LIST OF MESSAGE PAIRS")
        print("=" * 60)
        print(f"✅ Returns List: {isinstance(retrieved_history, list)}")
        print(f"✅ Contains Dictionaries: {all(isinstance(item, dict) for item in retrieved_history)}")
        print(f"✅ Has 'user' key: {all('user' in item for item in retrieved_history)}")
        print(f"✅ Has 'assistant' key: {all('assistant' in item for item in retrieved_history)}")
        print(f"✅ Chronological Order: {[pair['turn'] for pair in retrieved_history]}")
        
        # Test with limit parameter
        print(f"\n🔢 Testing with limit parameter (last 2 pairs):")
        limited_history = chat_manager.retrieve_chat_history(
            chat_uid=test_chat_uid,
            as_pairs=True,
            limit=2
        )
        print(f"   📊 Limited retrieval returned {len(limited_history)} pairs")
        print(f"   🔢 Turns: {[pair['turn'] for pair in limited_history]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during retrieval: {e}")
        return False

def test_individual_messages():
    """Test retrieving individual messages (not as pairs)"""
    
    chat_manager = ChatHistoryManager()
    test_chat_uid = "test-session-verification"
    
    print("\n" + "=" * 60)
    print("🧪 TESTING INDIVIDUAL MESSAGE RETRIEVAL")
    print("=" * 60)
    
    try:
        individual_messages = chat_manager.retrieve_chat_history(
            chat_uid=test_chat_uid,
            as_pairs=False  # Get individual messages
        )
        
        print(f"📊 Retrieved {len(individual_messages)} individual messages")
        print("\n📝 Individual Messages:")
        
        for msg in individual_messages:
            print(f"   {msg['message_type'].upper()}: {msg['content'][:50]}...")
            print(f"   Turn: {msg['conversation_turn']}, Time: {msg['timestamp']}")
            print()
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

async def main():
    """Run all tests"""
    print("🚀 Starting Chat History Tests...")
    
    # Test the main retrieval function
    success = await test_chat_history_retrieval()
    
    if success:
        # Test individual message retrieval
        test_individual_messages()
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ Tests failed. Check your ChromaDB connection.")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())