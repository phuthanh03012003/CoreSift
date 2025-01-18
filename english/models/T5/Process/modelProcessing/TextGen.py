import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

class TextGeneration:
    
    def __init__(self, model_name="t5-large"):
        # Tải mô hình và tokenizer T5
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def load_inputs(self, input_path):
        # Tải dữ liệu từ Decoder Layer (BaseModelOutput từ Encoder)
        self.inputs = torch.load(input_path)

        if isinstance(self.inputs, dict) and 'encoder_outputs' in self.inputs:
            print(" Đã tải dữ liệu từ Encoder Layer (encoder_outputs, attention_mask, decoder_input_ids).")
            self.encoder_outputs = self.inputs['encoder_outputs']
            self.attention_mask = self.inputs['attention_mask']

            print("🔸 Encoder Hidden States:", self.encoder_outputs.last_hidden_state[0, :7])
            print("🔸 Attention Mask:", self.attention_mask[0, :7])
        else:
            raise ValueError("Dữ liệu không đúng định dạng. Cần chứa 'encoder_outputs'.")

    def _prepare_inputs(self):
        # Chuẩn bị dữ liệu trước khi sinh văn bản
        batch_size, seq_len, _ = self.encoder_outputs.last_hidden_state.shape

        if self.attention_mask.shape[0] != batch_size:
            repeat_factor = batch_size // self.attention_mask.shape[0]
            self.attention_mask = self.attention_mask.repeat(repeat_factor, 1)

        self.attention_mask = self.attention_mask.to(self.model.device)
        self.encoder_outputs = BaseModelOutput(
            last_hidden_state=self.encoder_outputs.last_hidden_state.to(self.model.device)
        )

    def generate_text(self, strategy="beam_search"):
        """
        Sinh văn bản theo chiến lược tối ưu hóa:
        - "default": Beam Search mặc định (num_beams=5)
        - "greedy": Greedy Search (num_beams=1)
        - "beam_search": Beam Search (num_beams=1)
        - "top_k": Top-k Sampling (top_k=10)
        - "top_p": Top-p Sampling (top_p=0.5)
        """
        self._prepare_inputs()

        generation_config = {
            "encoder_outputs": self.encoder_outputs,
            "attention_mask": self.attention_mask,
            "max_length": 256,
            "min_length": 64,
            "length_penalty": 2.0,
        }

        file_name = "../object/TextGeneration/generated_"

        if strategy == "greedy":
            generation_config["num_beams"] = 1
            file_name += "greedy.pt"

        elif strategy == "beam_search":
            generation_config["num_beams"] = 1
            file_name += "beam_search.pt"

        elif strategy == "top_k":
            generation_config.update({"do_sample": True, "top_k": 10})
            generation_config["early_stopping"] = True
            file_name += "top_k.pt"

        elif strategy == "top_p":
            generation_config.update({"do_sample": True, "top_p": 0.5})
            generation_config["early_stopping"] = True
            file_name += "top_p.pt"

        else:
            file_name += "default.pt"

        # Sinh văn bản
        with torch.no_grad():
            generated_ids = self.model.generate(**generation_config)

        torch.save(generated_ids, file_name)
        print(f"📂 Token IDs đã được lưu vào {file_name}")

        strategy_name = {
            "default": "Mặc định - Beam Search",
            "greedy": "Greedy Search",
            "beam_search": "Beam Search (num_beams=1)",
            "top_k": "Top-k Sampling",
            "top_p": "Top-p Sampling"
        }
        print(f"\nToken IDs sinh ra ({strategy_name[strategy]}):\n{generated_ids[0]}")

def show_menu():
    print("\n===== CHỌN PHƯƠNG PHÁP SINH VĂN BẢN =====")
    print("1️. Mặc định (Beam Search)")
    print("2️. Greedy Search")
    print("3️. Beam Search (num_beams=1)")
    print("4️. Top-k Sampling")
    print("5️. Top-p Sampling")
    print("Nhấn phím bất kỳ để thoát")
    print("=========================================")

if __name__ == "__main__":
    text_gen = TextGeneration()
    text_gen.load_inputs("../object/decoder_outputs_default.pt")

    while True:
        show_menu()
        choice = input("Nhập lựa chọn (1-5): ")

        if choice == "1":
            text_gen.generate_text(strategy="default")
        elif choice == "2":
            text_gen.generate_text(strategy="greedy")
        elif choice == "3":
            text_gen.generate_text(strategy="beam_search")
        elif choice == "4":
            text_gen.generate_text(strategy="top_k")
        elif choice == "5":
            text_gen.generate_text(strategy="top_p")
        else:
            print("Thoát chương trình. Tạm biệt!")
            break
