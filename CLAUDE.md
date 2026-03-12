# CLAUDE.md

## Project Overview

**Kaggle Agent (KagGooLY)** — an AI-powered autonomous Kaggle problem solver using a "plan and execute" multi-agent architecture built on LangGraph. It orchestrates LLM agents to scrape challenge details, analyze datasets, plan tasks, generate code, execute it in Jupyter, and iterate with reflection-based error recovery.

## Repository Structure

```
kaggle-Agent/
├── main.py                     # Primary entry point — loads env, creates DI container, runs agent
├── agent.py                    # KaggleProblemSolver — LangGraph orchestrator (also CLI entry)
├── app.py                      # Streamlit web interface
├── config.ini                  # Application configuration (API, Kaggle, Jupyter, MongoDB)
├── config_reader.py            # INI configuration parser
├── di_container.py             # Dependency injection setup (Injector library)
│
├── planner_agent.py            # KaggleProblemPlanner — generates structured task plans
├── task_enhancer.py            # KaggleTaskEnhancer — enhances tasks with CoT reasoning
├── code_generation_agent.py    # CodeGenerationAgent — code gen with reflection loop (LangGraph subgraph)
├── replanner.py                # Re-planning fallback for failed tasks
├── kaggle_scraper.py           # ScrapeKaggle — Selenium-based challenge scraper
├── data_utils.py               # Dataset analysis utility (shape, types, missing values via LLM)
├── utils.py                    # Shared utilities (notebook execution interface)
│
├── states/                     # Pydantic state models
│   ├── main.py                 # KaggleProblemState (primary state)
│   ├── code.py                 # Code model
│   ├── enhancer.py             # EnhancedTask model
│   ├── memory.py               # MemoryAgent & WeightedMemory (short-term + RAG long-term)
│   └── write_challenge_docs.py # Challenge documentation template
│
├── executors/                  # Code execution engines
│   ├── nbexecutor_jupyter.py   # Jupyter kernel-based executor (primary)
│   ├── nbexecutor.py           # Base notebook executor
│   ├── nbexecutor_autoGen.py   # AutoGen-based executor
│   └── nbexecuter_e2b.py       # E2B sandbox executor
│
├── prompts/                    # LLM prompt templates
│   ├── prompts.py              # Planner prompt
│   ├── code_generation_prompt.py
│   ├── task_enhancer.py        # Task enhancement prompt
│   ├── summarizer_prompt.py    # Evaluation & summarization prompts
│   ├── utils.py                # Dataset analysis prompt
│   └── react.py                # ReAct pattern prompt
│
├── persistence/                # Data persistence layer
│   ├── mongo.py                # MongoDB checkpoint saver
│   └── postgres.py             # PostgreSQL connection
│
├── logging_module/             # Logging utilities
│   ├── logging.py              # Logger setup
│   └── log_it.py               # Log transformation
│
├── visualization/              # Graph visualization
│   └── vis.py
│
├── submission/                 # Kaggle submission handling
│   └── submission.py
│
├── examples/                   # Example challenge implementations
├── docs/                       # Project documentation
├── pyproject.toml              # Poetry project config (Python 3.12+)
├── requirements.txt            # Pip dependencies
├── Dockerfile                  # Docker image (base: jupyter/scipy-notebook)
├── docker-compose.yml          # Services: MongoDB, PostgreSQL, Langfuse, Jupyter
└── .pre-commit-config.yaml     # Pre-commit hooks config
```

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: Poetry
- **Agent Framework**: LangGraph (StateGraph-based orchestration)
- **LLM**: OpenAI GPT-4o via LangChain (`langchain-openai`)
- **Vector DB**: Chroma (RAG memory), FAISS (vector search)
- **Databases**: MongoDB (data/checkpoints), PostgreSQL (LangGraph checkpoints)
- **Notebook Execution**: Jupyter client/kernel
- **Web Scraping**: Selenium
- **Web UI**: Streamlit
- **Observability**: Langfuse
- **DI**: `injector` library

## Development Setup

```bash
# 1. Start services
docker-compose up -d

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.template .env   # Fill in API keys, DB connections, Kaggle creds

# 4. Run
poetry run python main.py
# or CLI: python agent.py --url <kaggle_url> [--cached <bool>]
# or Web: streamlit run app.py
```

## Build & Quality Commands

```bash
poetry install              # Install all dependencies
poetry run python main.py   # Run the agent

# Pre-commit hooks (run automatically on commit)
pre-commit run --all-files  # Run all hooks manually
```

There is no dedicated test suite. Testing is done via example challenges (e.g., `examples/titanic-spaceship/`).

## Code Style & Conventions

### Formatting & Linting
- **Ruff** for linting (`--fix` auto-fix enabled) and formatting
- **Black** as additional formatter
- **Pre-commit hooks** enforce: YAML/TOML/JSON validation, trailing whitespace, no debug statements, file size limit (1MB), docstring-first checks

### Commit Message Convention
Enforced by pre-commit hook — must follow conventional commits:
```
<type>(<scope>): <description>
```
Valid types: `break`, `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, `test`, `ops`, `hotfix`, `release`, `maint`, `init`, `enh`, `revert`

Example: `feat(agent): add memory persistence to code generation`

### Naming
- **Classes**: PascalCase (e.g., `KaggleProblemPlanner`, `EnhancedTask`)
- **Functions/variables**: snake_case
- **State models**: Pydantic `BaseModel` subclasses in `states/`
- **Prompts**: String constants in `prompts/` modules

### Architecture Patterns
- **Dependency Injection** via `injector` library (`di_container.py`)
- **State Management** via Pydantic models passed through LangGraph nodes
- **Graph-Based Orchestration** — LangGraph `StateGraph` with conditional edges
- **Reflection Loop** — code generation agent retries on error with LLM reflection
- **RAG Memory** — Chroma vector DB for long-term, weighted list for short-term

## Agent Workflow

```
START → Scraper → DataUtils → Planner → [Loop: Enhancer → CodeGenAgent → Executor] → END
```

The **CodeGenerationAgent** is itself a LangGraph subgraph:
```
Generate Code → Execute → Error? → (yes) Reflect → Generate Code (retry)
                                  → (no)  Return result
```

## Configuration

- **`config.ini`**: General settings (recursion limit, API model/temp, Kaggle URL, Jupyter, MongoDB)
- **`.env`**: Secrets (API keys, DB connection strings, Kaggle credentials, proxy settings)
- **`docker-compose.yml`**: Service ports — MongoDB:27017, PostgreSQL:5432, Langfuse:3000, Jupyter:8888

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Modify agent workflow | `agent.py` |
| Change planning logic | `planner_agent.py`, `prompts/prompts.py` |
| Modify code generation | `code_generation_agent.py`, `prompts/code_generation_prompt.py` |
| Update task enhancement | `task_enhancer.py`, `prompts/task_enhancer.py` |
| Change execution engine | `executors/nbexecutor_jupyter.py` |
| Update state schema | `states/main.py`, `states/code.py`, `states/enhancer.py` |
| Modify memory/RAG | `states/memory.py` |
| Change DI bindings | `di_container.py` |
| Add new prompts | `prompts/` directory |
| Modify scraping | `kaggle_scraper.py` |
