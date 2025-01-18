import torch
from transformers import T5Tokenizer
import re

class PostProcessing:
    def __init__(self, model_name="t5-large"):
        # Tải tokenizer của T5 để giải mã token thành văn bản
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def load_generated_tokens(self, token_path):
        # Tải token IDs đã sinh ra từ quá trình sinh văn bản
        self.generated_tokens = torch.load(token_path)
        print(f"\nĐã tải Token IDs từ {token_path}.")

    def decode_tokens(self):
        # Giải mã token thành văn bản tự nhiên
        decoded_text = self.tokenizer.decode(
            self.generated_tokens[0], skip_special_tokens=True
        )
        return decoded_text

    def clean_text(self, text):
        # 1️ Thay thế <n> hoặc </n> bằng dấu xuống dòng
        text = text.replace("<n>", "\n").replace("</n>", "\n")

        # 2️ Tự động xuống dòng sau dấu chấm, dấu hỏi, dấu chấm than
        text = re.sub(r'([.!?])\s+', r'\1\n', text)

        # 3️ Xóa khoảng trắng thừa và ký tự đặc biệt
        text = text.strip().replace("  ", " ")

        return text

    def save_final_output(self, text, save_path):
        # Lưu văn bản hoàn chỉnh vào file
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Đã lưu văn bản hoàn chỉnh vào {save_path}")

    def process_and_display(self, token_path, save_path, strategy_name):
        # Quy trình: Load -> Giải mã -> Làm sạch -> Lưu file -> In kết quả
        self.load_generated_tokens(token_path)
        decoded_text = self.decode_tokens()
        cleaned_text = self.clean_text(decoded_text)
        self.save_final_output(cleaned_text, save_path)

        # In kết quả ra màn hình
        print(f"\n📄 {strategy_name}:")
        print(cleaned_text)
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
        "3️. Beam Search (num_beams=10)": {
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
        post_processor.process_and_display(
            token_path=paths["token_path"],
            save_path=paths["save_path"],
            strategy_name=strategy_name
        )

    print("\n ĐÃ HOÀN THÀNH XỬ LÝ TẤT CẢ CÁC PHƯƠNG PHÁP.\n")
