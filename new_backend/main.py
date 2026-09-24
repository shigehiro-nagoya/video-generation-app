"""起動エントリポイント。

    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

その後 http://127.0.0.1:8000/docs で Swagger UI から実際にリクエストを試せます。
"""
from app.api import app  # noqa: F401
