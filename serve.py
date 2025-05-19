from waitress import serve
from app import app  # Assuming your file is named app.py

serve(app, host="0.0.0.0", port=8080)
