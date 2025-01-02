from transformers import pipeline

# Khởi tạo mô hình BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Đọc nội dung từ file văn bản
with open("Text.txt", "r") as file:
    text = file.read()

# Chia văn bản thành các đoạn nhỏ để xử lý
chunk_size = 1000
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Tóm tắt từng đoạn
summaries = [
    summarizer(chunk, max_length=130, min_length=50, do_sample=False)[0]['summary_text']
    for chunk in chunks
]

# Kết quả cuối cùng
final_summary = " ".join(summaries)

# Lưu vào file
with open("summarized_text.txt", "w") as output_file:
    output_file.write(final_summary)

print("Tóm tắt đã được lưu vào 'summarized_text.txt'")
