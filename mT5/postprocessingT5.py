import torch
from transformers import MT5Tokenizer
import re

class PostProcessing:
    def __init__(self, model_name="google/mt5-small"):
        # Load tokenizer để giải mã token thành văn bản
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)

    def load_generated_tokens(self, token_path):
        # Load token IDs từ quá trình sinh văn bản
        self.generated_tokens = torch.load(token_path)
        print(f"\nĐã tải Token IDs từ {token_path}.")

    def decode_tokens(self):
        # Giải mã token thành văn bản tự nhiên
        decoded_texts = [
            self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for tokens in self.generated_tokens
        ]
        return decoded_texts

    def clean_text(self, text):
        # 1️ Xóa khoảng trắng thừa và ký tự đặc biệt
        text = re.sub(r"\s+", " ", text).strip()
        # 2️ Chuẩn hóa cách viết hoa đầu câu
        text = text.capitalize()
        return text

    def save_final_output(self, texts, save_path):
        # Lưu danh sách văn bản ra file
        with open(save_path, "w", encoding="utf-8") as file:
            for i, text in enumerate(texts):
                file.write(f"Đoạn {i + 1}: {text}\n")
        print(f"Đã lưu văn bản hoàn chỉnh vào {save_path}")

    def process_and_display(self, token_path, save_path, strategy_name):
        # Xử lý từng file: Load -> Giải mã -> Làm sạch -> Lưu file -> In kết quả
        self.load_generated_tokens(token_path)
        decoded_texts = self.decode_tokens()
        cleaned_texts = [self.clean_text(text) for text in decoded_texts]
        self.save_final_output(cleaned_texts, save_path)

        # In kết quả ra màn hình
        print(f"\n{strategy_name}:")
        for i, text in enumerate(cleaned_texts[:5]):  # Hiển thị 5 kết quả đầu tiên
            print(f"Đoạn {i + 1}: {text}")
        print("=" * 80)

if __name__ == "__main__":
    post_processor = PostProcessing()

    # Mapping giữa file sinh token và file kết quả
    strategy_files = {
        "1️. Mặc định (Beam Search)": {
            "token_path": "object/TextGeneration/generated_default.pt",
            "save_path": "object/Outputs/final_output_default.txt"
        },
        "2️. Greedy Search": {
            "token_path": "object/TextGeneration/generated_greedy.pt",
            "save_path": "object/Outputs/final_output_greedy.txt"
        },
        "3️. Beam Search (num_beams=1)": {
            "token_path": "object/TextGeneration/generated_beam_search.pt",
            "save_path": "object/Outputs/final_output_beam_search.txt"
        },
        "4️. Top-k Sampling": {
            "token_path": "object/TextGeneration/generated_top_k.pt",
            "save_path": "object/Outputs/final_output_top_k.txt"
        },
        "5️. Top-p Sampling": {
            "token_path": "object/TextGeneration/generated_top_p.pt",
            "save_path": "object/Outputs/final_output_top_p.txt"
        }
    }

    # Xử lý toàn bộ các phương pháp
    for strategy_name, paths in strategy_files.items():
        if torch.load(paths["token_path"]) is not None:
            post_processor.process_and_display(
                token_path=paths["token_path"],
                save_path=paths["save_path"],
                strategy_name=strategy_name
            )

    print("\n ĐÃ HOÀN THÀNH XỬ LÝ TẤT CẢ CÁC PHƯƠNG PHÁP.\n")
