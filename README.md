# AI-Powered Resume Screening & Job Recommendation (Advanced)

This repository is a polished, interview-ready implementation of a Resume Screening
and Job Recommendation prototype. It focuses on clean code, robustness, and
production-minded design while remaining runnable in Google Colab for demo purposes.

Key features:
- Robust resume parsing (PDF / DOCX) with error handling and logs
- Hybrid matching: BERT embeddings + TF-IDF, configurable blending
- Keyword boosting (explicit skill matches increase score)
- SQLite storage with a small DB wrapper
- Flask API (ngrok-friendly for Colab demos)
- Unit tests (pytest) for core components
- Clear configuration and logging

See `demo_colab_steps.md` for quick Colab instructions.
