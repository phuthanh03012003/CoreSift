from transformers import AutoTokenizer, AutoModel
import torch
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Hàm tính embedding
def get_embedding(text):
    tokens = word_tokenize(text, format="text")
    input_ids = tokenizer.encode(tokens, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings.mean(dim=0).numpy()

# Hàm trích xuất từ khóa
def extract_keywords(text, top_n=5):
    words = list(set(word_tokenize(text, format="text").split()))
    text_embedding = get_embedding(text)
    word_embeddings = [get_embedding(word) for word in words]
    
    similarities = [cosine_similarity([text_embedding], [word_emb])[0][0] for word_emb in word_embeddings]
    keywords = [word for _, word in sorted(zip(similarities, words), reverse=True)]
    return keywords[:top_n]

# Đoạn code chính được bao bọc trong if __name__ == "__main__":
if __name__ == "__main__":
    # Tải mô hình PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2")
    
    # Thử nghiệm với văn bản mẫu
    text_sample = "Trí tuệ nhân tạo đang thay đổi cách chúng ta sống và làm việc hàng ngày."
    keywords = extract_keywords(text_sample, top_n=5)
    print("Từ khóa trích xuất:", keywords)