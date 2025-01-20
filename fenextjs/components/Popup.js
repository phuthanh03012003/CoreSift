import { useState } from "react";

export default function Popup() {
  const [inputText, setInputText] = useState("");
  const [summary, setSummary] = useState("");

  const handleSummarize = () => {
    if (!inputText.trim()) {
      alert("Please enter some text to test the UI!");
      return;
    }

    // Fake summary for testing
    const fakeSummary = "This is a simulated summary of your input.";
    setSummary(fakeSummary);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Test Popup</h1>
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter text here..."
        rows={5}
        style={styles.textarea}
      />
      <button onClick={handleSummarize} style={styles.button}>
        Summarize
      </button>
      {summary && (
        <div style={styles.summary}>
          <h3>Summary:</h3>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "20px",
    width: "300px",
    fontFamily: "Arial, sans-serif",
  },
  title: {
    fontSize: "18px",
    marginBottom: "10px",
  },
  textarea: {
    width: "100%",
    padding: "10px",
    marginBottom: "10px",
    fontSize: "14px",
    borderRadius: "5px",
    border: "1px solid #ddd",
  },
  button: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#007BFF",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    fontSize: "16px",
  },
  summary: {
    marginTop: "20px",
    backgroundColor: "#f9f9f9",
    padding: "10px",
    borderRadius: "5px",
    border: "1px solid #ddd",
  },
};
