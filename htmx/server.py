from flask import Flask, send_from_directory

app = Flask(__name__)

# Endpoint trả về file index.html khi truy cập "/"
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

# Endpoint để xử lý yêu cầu HTMX
@app.route('/example-endpoint')
def example_endpoint():
    # Trả về dữ liệu mẫu cho HTMX
    return '<p>Dữ liệu từ Flask server!</p>'

if __name__ == '__main__':
    app.run(port=8000)
