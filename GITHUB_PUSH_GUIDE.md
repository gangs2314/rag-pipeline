# GitHub Push Guide - RAG Pipeline

## What to Push vs What NOT to Push

### ✅ PUSH TO GITHUB (Version Controlled)

**Source Code:**
- `rag_pipeline/` - Complete Python package
- `tests/` - Test suite
- `Dockerfile` - Container build file
- `docker-compose.yml` - Deployment orchestration
- `nginx.conf` - Web server configuration
- `index.html` - Frontend UI
- `pyproject.toml` - Python dependencies
- `.gitignore` - Git ignore rules
- `.env.example` - Environment template (NO SECRETS)
- `README.md` - Main documentation

**Configuration:**
- `.claude/` - Claude Code configuration

### ❌ DO NOT PUSH TO GITHUB (Ignored by .gitignore)

**Data & Models (Large files):**
- `data/` - Vector store and uploaded documents (88MB+)
- `models/` - Cached embedding models (Large)
- `rag_pipeline.egg-info/` - Build artifacts

**Environment & Secrets:**
- `.env` - Never commit environment variables with secrets
- `certs/` - SSL certificates (security sensitive)

**OS & IDE:**
- `.DS_Store` - macOS files
- `__pycache__/` - Python cache
- `.vscode/`, `.idea/` - IDE settings

---

## Step-by-Step GitHub Push Instructions

### Step 1: Initialize Git Repository

```bash
cd /g/RAG
git init
```

**Expected output:**
```
Initialized empty Git repository in /g/RAG/.git/
```

---

### Step 2: Configure Git User (If First Time)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or for this repo only (without `--global`):
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

### Step 3: Add All Files to Staging Area

```bash
git add .
```

**Verify what will be committed:**
```bash
git status
```

**Expected output:**
```
On branch master

No commits yet

Changes to be committed:
  new file:   .env.example
  new file:   .gitignore
  new file:   Dockerfile
  new file:   README.md
  new file:   docker-compose.yml
  new file:   index.html
  new file:   nginx.conf
  new file:   pyproject.toml
  new file:   rag_pipeline/...
  ...
```

---

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: RAG Pipeline with advanced deduplication

- Complete RAG pipeline with document ingestion and semantic search
- FastAPI REST API with CORS support
- Advanced deduplication system (exact hashing + n-gram similarity)
- ChromaDB vector store integration
- Docker containerization with nginx frontend
- Production-ready with comprehensive error handling
- Support for multiple document formats (PDF, TXT, DOCX, etc.)
- Parent-child hierarchical chunking
- Interactive web UI for document upload and search"
```

---

### Step 5: Create GitHub Repository

1. Go to https://github.com/new
2. **Repository name**: `rag-pipeline` (or your choice)
3. **Description**: "Production-grade Retrieval-Augmented Generation system with advanced deduplication"
4. **Visibility**: Choose `Public` or `Private`
5. **Initialize repository**: ❌ Do NOT initialize with README (we have one)
6. Click "Create repository"

---

### Step 6: Add Remote Repository

After creating the GitHub repo, you'll see a command like:

```bash
git remote add origin https://github.com/YOUR_USERNAME/rag-pipeline.git
```

Or with SSH (if you have SSH key set up):

```bash
git remote add origin git@github.com:YOUR_USERNAME/rag-pipeline.git
```

**Verify the remote:**
```bash
git remote -v
```

**Expected output:**
```
origin  https://github.com/YOUR_USERNAME/rag-pipeline.git (fetch)
origin  https://github.com/YOUR_USERNAME/rag-pipeline.git (push)
```

---

### Step 7: Rename Branch (Optional but Recommended)

GitHub uses `main` as default (not `master`):

```bash
git branch -M main
```

---

### Step 8: Push to GitHub

**First push (sets upstream):**
```bash
git push -u origin main
```

**For subsequent pushes:**
```bash
git push
```

**Expected output:**
```
Enumerating objects: 150, done.
Counting objects: 100% (150/150), done.
Delta compression using up to 8 threads
Compressing objects: 100% (120/120), done.
Writing objects: 100% (150/150), 2.45 MiB | 1.23 MiB/s, done.
Total 150 (delta 45), reused 0 (delta 0), reused pack 0
remote: Resolving deltas: 100% (45/45), done.
To https://github.com/YOUR_USERNAME/rag-pipeline.git
 * [new branch]      main -> main
```

---

## Complete Command Sequence

**Copy and paste this entire block:**

```bash
cd /g/RAG

# Initialize git
git init

# Configure git (if needed)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Stage all files
git add .

# Verify what will be committed
git status

# Create initial commit
git commit -m "Initial commit: RAG Pipeline with advanced deduplication

- Complete RAG pipeline with document ingestion and semantic search
- FastAPI REST API with CORS support
- Advanced deduplication system (exact hashing + n-gram similarity)
- ChromaDB vector store integration
- Docker containerization with nginx frontend
- Production-ready with comprehensive error handling
- Support for multiple document formats (PDF, TXT, DOCX, etc.)
- Parent-child hierarchical chunking
- Interactive web UI for document upload and search"

# Rename to main branch
git branch -M main

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/rag-pipeline.git

# Verify remote
git remote -v

# Push to GitHub
git push -u origin main
```

---

## What Gets Committed (File Count & Size)

**Estimated files to be committed:**
- ~50 Python files (rag_pipeline/ package)
- 1 HTML file (frontend)
- 3 Config files (Docker, nginx, pyproject.toml)
- 1 README
- Test files

**Total size**: ~3-5 MB (much smaller because data/ and models/ are ignored)

**NOT committed** (~100MB):
- data/ directory (ignored)
- models/ directory (ignored)
- __pycache__ directories (ignored)

---

## After Pushing: Next Steps

### 1. Add a License

```bash
# Create LICENSE file (MIT recommended)
# Then:
git add LICENSE
git commit -m "Add MIT license"
git push
```

### 2. Create Issues for Future Work

Use GitHub Issues to track:
- Bug fixes
- Feature requests
- Performance improvements

### 3. Set Up GitHub Pages (Optional)

If you want documentation website:
- Settings → Pages → Source: main branch /docs folder

### 4. Enable Actions (Optional)

Set up CI/CD for automated testing:
- Add `.github/workflows/` with test workflows

---

## Updating the Repository Later

### Add New Changes

```bash
# Make changes to files
# Then:
git add .
git commit -m "Description of changes"
git push
```

### Update Without Adding New Files

```bash
git add -A
git commit -m "Your message"
git push
```

### View History

```bash
git log --oneline
```

### Create a New Branch for Features

```bash
git checkout -b feature/amazing-feature
# Make changes
git add .
git commit -m "Add amazing feature"
git push -u origin feature/amazing-feature
# Then create Pull Request on GitHub
```

---

## Troubleshooting

### Authentication Error

**Problem:** `fatal: Authentication failed`

**Solution:**
- HTTPS: Use GitHub Personal Access Token instead of password
  - Go to: Settings → Developer settings → Personal access tokens
  - Generate token with `repo` scope
  - Use token as password

- SSH: Add SSH key
  - Generate: `ssh-keygen -t ed25519`
  - Add public key to GitHub: Settings → SSH and GPG keys

### Rejected Push

**Problem:** `Updates were rejected because...`

**Solution:**
```bash
git pull origin main
# Resolve conflicts if any
git add .
git commit -m "Merge changes"
git push
```

### Wrong Commit Message

```bash
# Edit last commit message (before push)
git commit --amend -m "New message"
git push --force-with-lease

# Or create a new commit (after push)
git commit -m "Fix: corrected information"
git push
```

---

## Verification Checklist

Before pushing, verify:

- [ ] `git status` shows clean working directory
- [ ] `.gitignore` properly ignores data/, models/
- [ ] No `.env` file with secrets in staging
- [ ] `README.md` is present and up-to-date
- [ ] `pyproject.toml` has correct dependencies
- [ ] `docker-compose.yml` is present
- [ ] `.env.example` has no actual secrets
- [ ] Commit message is descriptive
- [ ] Remote URL is correct
- [ ] Pushing to correct branch (main)

---

## File Structure After Push

Your GitHub repository will contain:

```
rag-pipeline/
├── rag_pipeline/          # Source code
│   ├── api.py
│   ├── pipeline.py
│   ├── deduplication.py
│   ├── vector_store.py
│   ├── chunking.py
│   ├── loaders.py
│   ├── config.py
│   ├── cli.py
│   └── ...
├── tests/                 # Test suite
├── Dockerfile             # Container build
├── docker-compose.yml     # Deployment config
├── nginx.conf             # Web server config
├── index.html             # Frontend UI
├── pyproject.toml         # Dependencies
├── .gitignore             # Git ignore rules
├── .env.example           # Environment template
├── README.md              # Documentation
└── GITHUB_PUSH_GUIDE.md   # This file
```

**NOT included (ignored):**
- data/ (vector store)
- models/ (cached embeddings)
- __pycache__/
- .env (actual secrets)

---

## Size Expectations

| Item | Size | Included |
|------|------|----------|
| Source code | ~150 KB | ✅ Yes |
| Documentation | ~50 KB | ✅ Yes |
| Config files | ~20 KB | ✅ Yes |
| Frontend (index.html) | ~30 KB | ✅ Yes |
| **Total to push** | **~3-5 MB** | **✅ Yes** |
| data/ directory | 88+ MB | ❌ No (.gitignored) |
| models/ directory | Large | ❌ No (.gitignored) |

---

## Quick Reference Commands

```bash
# View all changes
git diff

# See commit history
git log --oneline

# Undo last commit (before push)
git reset --soft HEAD~1

# Check what will be pushed
git log origin/main..HEAD

# See remote info
git remote show origin

# Create a tag
git tag v0.1.0
git push origin v0.1.0
```

---

**Ready to push? Follow the "Complete Command Sequence" section above!**
