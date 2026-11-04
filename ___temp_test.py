import chromadb

# ساخت دیتابیس محلی
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# ساخت یک مجموعه برای خاطره‌ها
collection = client.get_or_create_collection(
    name="agent_memories"
)

# ذخیره چند خاطره
collection.add(
    ids=["memory_1", "memory_2", "memory_3"],
    documents=[
        "The user knows Python programming.",
        "The user is building an AI Agent.",
        "The user uses Visual Studio Code."
    ]
)

# جست‌وجوی معنایی
results = collection.query(
    query_texts=[
        "What does the user know about programming?"
    ],
    n_results=2
)

print(results["documents"])