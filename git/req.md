# Deep Learning Lab Requirements

## Python Packages Required

### Core Machine Learning & Deep Learning Libraries
```bash
pip install torch torchvision torchaudio
pip install scikit-learn
pip install numpy
```

### Data Processing & Analysis
```bash
pip install pandas
pip install matplotlib
pip install seaborn
```

### Image Processing
```bash
pip install Pillow  # PIL
pip install opencv-python  # Optional for cv2 (used in useless/enhance.py)
```

### Standard Libraries (Built-in)
- `math` - Mathematical functions
- `random` - Random number generation
- `copy` - Object copying utilities

## VS Code Extensions Required

### Essential Python Development Extensions

#### 1. **Python** (ms-python.python)
- **Purpose**: Core Python language support, IntelliSense, linting, debugging
- **Features**: 
  - Syntax highlighting and code completion
  - Integrated terminal with Python REPL
  - Code formatting with autopep8, black, yapf
  - Linting with pylint, flake8, mypy
- **Installation**: `code --install-extension ms-python.python`
- **Configuration**: Automatically detects Python interpreters

#### 2. **Pylance** (ms-python.vscode-pylance)
- **Purpose**: Advanced Python language server with type checking
- **Features**:
  - Fast IntelliSense and auto-completion
  - Type checking and error detection
  - Import organization and unused import detection
  - Go to definition, find references
- **Installation**: `code --install-extension ms-python.vscode-pylance`
- **Settings**: Enable in Python › Analysis: Type Checking Mode

#### 3. **Python Debugger** (ms-python.debugpy)
- **Purpose**: Advanced debugging capabilities for Python
- **Features**:
  - Breakpoints and conditional breakpoints
  - Variable inspection and watch expressions
  - Call stack navigation
  - Debug console for live evaluation
- **Installation**: `code --install-extension ms-python.debugpy`

### Jupyter Notebook & Data Science Extensions

#### 4. **Jupyter** (ms-toolsai.jupyter)
- **Purpose**: Complete Jupyter notebook support in VS Code
- **Features**:
  - Create, edit, and run .ipynb files
  - Interactive Python execution
  - Rich output display (plots, HTML, markdown)
  - Kernel management and selection
- **Installation**: `code --install-extension ms-toolsai.jupyter`
- **Usage**: Essential for running notebook-style experiments

#### 5. **Jupyter Keymap** (ms-toolsai.jupyter-keymap)
- **Purpose**: Jupyter-style keyboard shortcuts
- **Features**:
  - Familiar Jupyter Lab/Notebook keybindings
  - Cell navigation (Ctrl+Enter, Shift+Enter)
  - Quick cell type switching (M for markdown, Y for code)
- **Installation**: `code --install-extension ms-toolsai.jupyter-keymap`

#### 6. **Jupyter Notebook Renderers** (ms-toolsai.jupyter-renderers)
- **Purpose**: Enhanced rendering for notebook outputs
- **Features**:
  - Better matplotlib plot rendering
  - Interactive widgets support
  - HTML and LaTeX rendering
  - Data visualization improvements
- **Installation**: `code --install-extension ms-toolsai.jupyter-renderers`

#### 7. **Jupyter Cell Tags** (ms-toolsai.vscode-jupyter-cell-tags)
- **Purpose**: Organize and manage notebook cells
- **Features**:
  - Tag cells for organization
  - Filter cells by tags
  - Export specific tagged cells
- **Installation**: `code --install-extension ms-toolsai.vscode-jupyter-cell-tags`

### Machine Learning & Data Science Specific

#### 8. **Python Docstring Generator** (njpwerner.autodocstring)
- **Purpose**: Auto-generate Python docstrings
- **Features**:
  - Multiple docstring formats (Google, NumPy, Sphinx)
  - Function signature analysis
  - Template customization
- **Installation**: `code --install-extension njpwerner.autodocstring`
- **Usage**: Type `"""` and press Tab to generate docstrings

#### 9. **Python Type Hint** (njqdev.vscode-python-typehint)
- **Purpose**: Enhanced type hinting support
- **Features**:
  - Auto-completion for type annotations
  - Type hint suggestions
  - Better mypy integration
- **Installation**: `code --install-extension njqdev.vscode-python-typehint`

### Code Quality & Formatting

#### 10. **Python Indent** (KevinRose.vsc-python-indent)
- **Purpose**: Correct Python indentation
- **Features**:
  - Smart indentation for Python syntax
  - Handles complex nested structures
  - Prevents common indentation errors
- **Installation**: `code --install-extension KevinRose.vsc-python-indent`

#### 11. **Black Formatter** (ms-python.black-formatter)
- **Purpose**: Code formatting with Black
- **Features**:
  - Consistent code style
  - Format on save option
  - PEP 8 compliance
- **Installation**: `code --install-extension ms-python.black-formatter`
- **Configuration**: Set as default formatter in settings

### Git & Version Control

#### 12. **GitLens** (eamodio.gitlens)
- **Purpose**: Supercharge Git capabilities
- **Features**:
  - Git blame annotations
  - Commit history visualization
  - Branch and repository insights
  - File history and comparisons
- **Installation**: `code --install-extension eamodio.gitlens`

#### 13. **Git Graph** (mhutchie.git-graph)
- **Purpose**: Visualize Git repository history
- **Features**:
  - Interactive commit graph
  - Branch visualization
  - Commit details and diffs
- **Installation**: `code --install-extension mhutchie.git-graph`

### Optional but Highly Recommended

#### 14. **Error Lens** (usernamehw.errorlens)
- **Purpose**: Highlight errors and warnings inline
- **Features**:
  - Inline error/warning display
  - Quick problem identification
  - Customizable error highlighting
- **Installation**: `code --install-extension usernamehw.errorlens`

#### 15. **Material Icon Theme** (PKief.material-icon-theme)
- **Purpose**: Better file and folder icons
- **Features**:
  - Language-specific icons
  - Clear visual file type identification
  - Customizable icon themes
- **Installation**: `code --install-extension PKief.material-icon-theme`

#### 16. **Thunder Client** (rangav.vscode-thunder-client)
- **Purpose**: REST API client (useful for ML API testing)
- **Features**:
  - Test REST APIs
  - Save and organize requests
  - Environment variables support
- **Installation**: `code --install-extension rangav.vscode-thunder-client`

### Extension Installation Commands

```bash
# Install all essential extensions at once
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension njpwerner.autodocstring
code --install-extension KevinRose.vsc-python-indent
code --install-extension ms-python.black-formatter
code --install-extension eamodio.gitlens
code --install-extension mhutchie.git-graph
code --install-extension usernamehw.errorlens
code --install-extension PKief.material-icon-theme
```

### Recommended VS Code Settings

Add these to your `settings.json` for optimal Python/ML development:

```json
{
    "python.defaultInterpreterPath": "./deeplearning_lab/Scripts/python.exe",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "jupyter.askForKernelRestart": false,
    "jupyter.interactiveWindowMode": "perFile",
    "files.autoSave": "afterDelay",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## Installation Commands

### Complete Package Installation
```bash
# Create virtual environment (recommended)
python -m venv deeplearning_lab
source deeplearning_lab/bin/activate  # On Windows: deeplearning_lab\Scripts\activate

# Install all required packages
pip install torch torchvision torchaudio scikit-learn numpy pandas matplotlib seaborn Pillow

# Optional: Install additional packages
pip install opencv-python jupyter
```

### Hardware Requirements
- **CUDA-capable GPU** (optional, for faster training)
- **Minimum 8GB RAM** (16GB recommended for larger models)
- **Python 3.8+** required

### Datasets Used
- **CIFAR-10** - Downloaded automatically via torchvision
- **Iris Dataset** - Included with scikit-learn
- **PlayTennis.csv** - Custom dataset in exp1/a/
- **Input images** - Custom images in exp3/ folder