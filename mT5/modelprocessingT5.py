import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Load model và tokenizer
model_name = "google/mt5-small"  # Bạn có thể chọn phiên bản khác như "mt5-base", "mt5-large" nếu cần
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# Load dữ liệu đã xử lý từ preprocessing step
input_file = "processed_mt5_inputs.pt"
inputs = torch.load(input_file)

# Hàm thực hiện suy luận và giải mã đầu ra
def generate_keywords(input_ids, attention_mask, num_beams=5, max_length=50):
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,  # Beam Search để cải thiện kết quả
        max_length=max_length,  # Giới hạn độ dài của đầu ra
        early_stopping=True
    )
    return outputs

# Lấy đầu vào từ tensor
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Thực hiện suy luận
generated_ids = generate_keywords(input_ids, attention_mask)

# Giải mã đầu ra
generated_keywords = [
    tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for g in generated_ids
]

# Lưu kết quả ra file
output_file = "generated_keywords.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, keywords in enumerate(generated_keywords):
        f.write(f"Đoạn {i + 1}: {keywords}\n")

print(f"Đã lưu kết quả trích xuất từ khóa vào file: {output_file}")

# Hiển thị một số kết quả đầu ra để kiểm tra
for i, keywords in enumerate(generated_keywords[:5]):  # Hiển thị 5 đoạn đầu tiên
    print(f"Đoạn {i + 1}: {keywords}")
