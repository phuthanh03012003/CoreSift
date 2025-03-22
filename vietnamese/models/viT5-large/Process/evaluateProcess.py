import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from textblob import TextBlob

class Evaluation:
    def __init__(self, output_dir="./object"):
        self.output_dir = output_dir
        self.reference_path = os.path.join("..", "..", "..", "inputs", "TamCam.txt")
        self.result_file = os.path.join(self.output_dir, "evaluation_results.txt")

        # Xóa nội dung cũ nếu file đã tồn tại
        with open(self.result_file, "w", encoding="utf-8") as file:
            file.write("**TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ**\n")
            file.write("=" * 80 + "\n")

    def load_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def compute_bleu(self, reference, generated):
        reference_tokens = reference.split()
        generated_tokens = generated.split()
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        return bleu_score

    def compute_rouge(self, reference, generated):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores

    def grammar_check(self, text):
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        return corrected_text

    def save_evaluation(self, strategy_name, bleu_score, rouge_scores, corrected_text):
        # Ghi kết quả đánh giá vào file
        with open(self.result_file, "a", encoding="utf-8") as file:
            file.write(f"\n**Đánh giá văn bản ({strategy_name}):**\n")
            file.write(f"🔹 **BLEU Score:** {bleu_score:.4f}\n")
            file.write(f"🔹 **ROUGE-1:** {rouge_scores['rouge1'].fmeasure:.4f}\n")
            file.write(f"🔹 **ROUGE-2:** {rouge_scores['rouge2'].fmeasure:.4f}\n")
            file.write(f"🔹 **ROUGE-L:** {rouge_scores['rougeL'].fmeasure:.4f}\n\n")
            file.write("**Văn bản sau khi sửa ngữ pháp:**\n")
            file.write(corrected_text + "\n")
            file.write("=" * 80 + "\n")

    def evaluate_text(self, strategy_name, file_path):
        reference_text = self.load_text(self.reference_path)
        generated_text = self.load_text(file_path)

        # Tính điểm BLEU
        bleu_score = self.compute_bleu(reference_text, generated_text)

        # Tính điểm ROUGE
        rouge_scores = self.compute_rouge(reference_text, generated_text)

        # Kiểm tra lỗi ngữ pháp
        corrected_text = self.grammar_check(generated_text)

        # In kết quả ra màn hình
        print(f"\n**Đánh giá văn bản ({strategy_name}):**")
        print(f"🔹 BLEU Score: {bleu_score:.4f}")
        print(f"🔹 ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}")
        print(f"🔹 ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}")
        print(f"🔹 ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}")
        print("\n** Văn bản sau khi sửa ngữ pháp:**")
        print(corrected_text)
        print("=" * 80)

        # Lưu kết quả vào file
        self.save_evaluation(strategy_name, bleu_score, rouge_scores, corrected_text)

    def evaluate_all_methods(self):
        output_dir = os.path.abspath(self.output_dir)

        methods = {
            "1️. Mặc định (Beam Search)": os.path.join(output_dir, "Outputs", "final_output_default.txt"),
            "2️. Greedy Search": os.path.join(output_dir, "Outputs", "final_output_greedy.txt"),
            "3️. Beam Search (num_beams=1)": os.path.join(output_dir, "Outputs", "final_output_beam_search.txt"),
            "4️. Top-k Sampling": os.path.join(output_dir, "Outputs", "final_output_top_k.txt"),
            "5️. Top-p Sampling": os.path.join(output_dir, "Outputs", "final_output_top_p.txt")
        }

        for strategy_name, file_path in methods.items():
            if not os.path.exists(file_path):
                print(f" File không tồn tại: {file_path}")
            else:
                self.evaluate_text(strategy_name, file_path)

if __name__ == "__main__":
    evaluator = Evaluation()
    evaluator.evaluate_all_methods()
    print("\n**Đã lưu kết quả đánh giá vào file:** object/evaluation_results.txt\n")
