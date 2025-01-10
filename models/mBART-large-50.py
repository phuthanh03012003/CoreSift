from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os

# Load model và tokenizer
model_name = "facebook/mbart-large-50"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Cấu hình ngôn ngữ nguồn và đích
source_language = "vi_VN"  # Tiếng Việt
target_language = "vi_VN"  # Tiếng Việt (tóm tắt hoặc tái diễn đạt)
tokenizer.src_lang = source_language

# Đọc nội dung từ file input
input_file = "../input.txt"
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

# Tokenize văn bản
inputs = tokenizer(raw_text, return_tensors="pt", max_length=1024, truncation=True)

# Generate output (tóm tắt hoặc tái diễn đạt)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=50,          # Độ dài tối đa của kết quả
    min_length=10,          # Độ dài tối thiểu của kết quả
    length_penalty=2.0,     # Hệ số phạt độ dài (càng lớn, càng ưu tiên văn bản ngắn hơn)
    num_beams=4,            # Beam search
    early_stopping=True     # Dừng sớm khi đạt kết quả tốt
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# In kết quả ra màn hình
print("Kết quả:", summary)

output_file = "../output/mBART-large-50_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Kết quả đã được ghi vào file: {output_file}")
