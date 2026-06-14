# Contributing to RAG-Sprint

Thank you for your interest in contributing!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-sprint.git
   cd rag-sprint
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Testing Your Changes

Run any Python file directly to test:
```bash
python src/chunk_text.py
python src/semantic_search.py
```

All files have built-in test blocks using `if __name__ == "__main__":`.

## Pull Request Process

1. Update documentation if needed (README.md, LEARNING_GUIDE.md)
2. Ensure your code has comments explaining the logic
3. Test your changes with the existing test blocks
4. Push to your fork and create a Pull Request
5. Describe what your changes do and why

## Code Style

- Use clear variable names
- Add comments explaining complex logic
- Follow the existing comment structure: "WHAT," "WHY," "HOW"
- Use type hints in function definitions

Thank you! 🙏
