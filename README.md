# İlk Yardım RAG Asistanı

Bu proje, acil durumlarda hızlı ve güvenilir bilgi sağlamak amacıyla geliştirilmiş, **RAG (Retrieval-Augmented Generation)** mimarisine sahip yapay zeka tabanlı bir ilk yardım asistanıdır.



##  Öne Çıkan Özellikler

* **Çok Kanallı Veri İşleme:** PDF, JSON ve TXT formatındaki belgeleri otomatik olarak okur ve indeksler.
* **Semantik Arama:** Kullanıcı sorularını sadece anahtar kelimelerle değil, anlamsal bağlamıyla (Sentence Transformers) anlar.
* **Gelişmiş Reranking:** Çekilen bilgileri `FlashRank` kullanarak en alakalı olanlara göre yeniden sıralar.
* **Güvenilir Yanıtlar:** Claude 3.5 (Haiku) modeli ile sadece bilgi tabanındaki verilere sadık kalarak yanıt üretir.
* **Kullanıcı Dostu Arayüz:** `Gradio` ile temiz ve hızlı bir web arayüzü sunar.
  

## Teknoloji Yığını (Tech Stack)

| Kategori | Teknoloji | Görevi |
| :--- | :--- | :--- |
| **LLM** | Anthropic Claude 3.5 | Yanıt üretimi ve akıl yürütme. |
| **Vector DB** | ChromaDB | Vektörel veri depolama ve hızlı retrieval. |
| **Embedding** | all-MiniLM-L6-v2 | Metinleri matematiksel vektörlere dönüştürme. |
| **Arayüz** | Gradio | Chatbot UI bileşenleri. |
| **Reranker** | FlashRank | Alınan verilerin doğruluğunu optimize etme. |



##  Kurulum

1. **Depoyu klonlayın:**
   ```bash
   git clone [https://github.com/Sacidea/Ilk-yardim-rag-asistan-.git](https://github.com/Sacidea/Ilk-yardim-rag-asistan-.git)
   cd Ilk-yardim-rag-asistan-

   <img width="2560" height="2674" alt="127 0 0 1_7860_ (1)" src="https://github.com/user-attachments/assets/8a92e9a9-48c1-4023-9b4a-def6d5027170" />
