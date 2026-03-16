import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

# ── 1. VERİ KATMANI ──
def load_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = []
    for block in content.split("\n\n"):
        block = block.strip()
        if block:
            chunks.append(block)
    return chunks

# ── 2. EMBEDDING MODELİ ──
print("Embedding modeli yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 3. VECTOR DATABASE ──
client = chromadb.Client()
collection = client.create_collection("ilk_yardim")

def build_index(chunks):
    print(f"{len(chunks)} chunk vektöre dönüştürülüyor...")
    embeddings = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Vector DB hazır!")

# ── 4. RETRIEVER ──
def retrieve(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

# ── 5. LLM ──
llm = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def ask(query, history):
    chunks = retrieve(query)
    context = "\n\n---\n\n".join(chunks)

    print(f"\n[Retriever] {len(chunks)} chunk bulundu.")

    system = f"""Sen bir Türkçe ilk yardım asistanısın.
Yalnızca aşağıdaki bilgi tabanını kullan. Bilgi tabanında yoksa 'Bilgi tabanımda bu konu yok' de.
Acil durumlarda mutlaka 112'yi hatırlat.

BİLGİ TABANI:
{context}"""

    history.append({"role": "user", "content": query})

    response = llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=history
    )

    answer = response.content[0].text
    history.append({"role": "assistant", "content": answer})
    return answer

# ── 6. ARAYÜZ ──
def main():
    print("Veri yükleniyor...")
    chunks = load_documents("data/ilk_yardim.txt")
    build_index(chunks)

    print("\n" + "="*50)
    print("   İLK YARDIM RAG CHATBOT")
    print("   Çıkmak için 'q' yaz")
    print("="*50 + "\n")

    history = []
    while True:
        query = input("Sen: ").strip()
        if query.lower() == "q":
            print("Güle güle!")
            break
        if not query:
            continue
        answer = ask(query, history)
        print(f"\nAsistan: {answer}\n")

if __name__ == "__main__":
    main()