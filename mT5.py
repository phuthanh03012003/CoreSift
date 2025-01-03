from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Tải mô hình và tokenizer mT5
model_name = "google/mt5-base"  # Sử dụng mT5-Base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Đọc nội dung từ file .txt
file_path = "test.txt"  # Đường dẫn đến file .txt của bạn
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Giới hạn đầu vào
max_input_length = 512  # Tokens tối đa mô hình hỗ trợ
max_words = 250  # Số từ tối đa trước khi chia nhỏ

# Hàm đếm từ
def count_words(text):
    return len(text.split())

# Hàm chia nhỏ văn bản
def split_text(text, max_words):
    words = text.split()
    chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks

# Kiểm tra số từ
if count_words(text) > max_words:
    print("Văn bản vượt quá số từ cho phép. Đang chia nhỏ...")
    chunks = split_text(text, max_words)
    
    summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"Đang xử lý đoạn {idx + 1}/{len(chunks)}...")
        inputs = tokenizer(
            "summarize: " + chunk, 
            return_tensors="pt", 
            max_length=max_input_length, 
            truncation=True
        )
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=150,  # Giới hạn độ dài tóm tắt mỗi đoạn
            num_beams=4, 
            early_stopping=True
        )
        # Lưu tóm tắt của từng đoạn
        summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Kết hợp tóm tắt
    final_summary = " ".join(summaries)
    print("\nTóm tắt tổng hợp:")
    print(final_summary)

else:
    print("Văn bản trong giới hạn số từ. Đang tóm tắt...")
    inputs = tokenizer(
        "summarize: " + text, 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    )
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=200,  # Giới hạn độ dài tóm tắt
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nTóm tắt văn bản:")
    print(summary)
