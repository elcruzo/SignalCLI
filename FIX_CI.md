# Fix CI Issues

The GitHub Actions failed due to formatting and dependency issues. Here's how to fix them:

## 1. Format Code with Black

Run this locally before pushing:

```bash
# Install black if not already installed
pip install black==23.11.0

# Format all code
black src tests

# Or use the provided script
python format.py
```

## 2. Dependencies Fixed

Already fixed in requirements.txt:
- Removed conflicting `langchain` and `langchain-community` 
- These weren't actually used in the code

## 3. CodeQL Version Fixed

Updated `.github/workflows/` files to use CodeQL v3 instead of v2.

## 4. Next Steps

1. Run Black locally to format all files
2. Commit the changes:
   ```bash
   git add .
   git commit -m "fix: format code with black and update dependencies"
   git push
   ```

## 5. Optional: Pre-commit Hook

To avoid this in the future, install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

This will automatically format code before each commit.