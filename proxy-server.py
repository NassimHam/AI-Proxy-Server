"""Minimal entrypoint: run the FastAPI app via uvicorn.

All routes and configuration have been moved to separate modules:
- `app.py` (FastAPI app + middleware)
- `handlers.py` (route handlers)
- `config.py`, `models.py`, `utils.py`

Keep this file tiny so it's easy to run or replace with other launchers.
"""

from app import app
import uvicorn



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)