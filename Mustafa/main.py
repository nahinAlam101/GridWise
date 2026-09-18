"""Run with `python main.py` or `uvicorn main:app`."""

import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from gridwise.app import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
