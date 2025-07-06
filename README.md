# 🚀 ArxivSearcher MCP Server

An MCP server for intelligent searching of Software Engineering papers on arXiv, with advanced filtering and sorting.

---

## 📝 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/emi-dm/Arxiv-MCP.git
   cd Arxiv-MCP
   ```


2. **Choose your setup method:**
   - [Using Docker (Recommended)](#option-1-using-docker-recommended)
   - [Local configuration with uv](#option-2-local-configuration-with-uv-requires-code-locally)
   - [New! Consume the MCP server directly from the cloud](#option-3-consume-the-mcp-server-in-the-cloud)

---

## 🖱️ Using with Cursor

If you use [Cursor](https://www.cursor.so/), you can integrate this MCP server as follows:

1. Open **Settings** in Cursor.
2. Go to **Tools & Integrations**.
3. In the **MCP** section, add a new server configuration.
4. Use the same configuration as described for VS Code:
   - For Docker, use the Docker configuration block from the VS Code section.
   - For local development, use the uv configuration block from the VS Code section.

The process is analogous to the VS Code setup—just copy the relevant configuration and paste it into Cursor's MCP server settings.

---

## 📋 Prerequisites

Before you begin, make sure you have installed:
- [Python](https://www.python.org/downloads/) (3.11 or higher)
- [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver)
- [Node.js and npm](https://nodejs.org/en/download/) (for debugging with MCP Inspector)

**Or alternatively:**
- [Docker](https://docs.docker.com/get-docker/) (for containerized deployment)

---



## ⚡️ Quickstart in VS Code

You can run the server in VS Code using Docker (recommended, you don't need the code locally), locally with uv, **or consume it directly from the cloud**.

### Option 3: Consume the MCP server in the cloud (Recommended for quick testing!)

You don't need to install anything or clone the repository. You can consume the MCP server directly using the following configuration in `.vscode/mcp.json` or in your compatible MCP client:

```json
{
  "servers": {
    "arxiv-mcp": {
      "url": "https://arxiv-mcp-sq0a.onrender.com/mcp/"
    }
  }
}
```

This works in both VS Code and [Cursor](https://www.cursor.so/) and other compatible MCP clients.


### Option 1: Using Docker (Recommended)
> **Note:** If you use Docker, you do not need to have the code in your local directory. The container will run everything needed.

1. **Build the Docker image (only needed once or when you update the code):**
   
   ```bash
   docker build -t arxiv-searcher-mcp .
   ```

2.  **Create `.vscode/mcp.json`:**
    In your project root, create the `.vscode` folder if it doesn't exist. Inside, create a file named `mcp.json`.

3.  **Add the Docker server configuration:**
    Copy and paste the following configuration into `.vscode/mcp.json`:

    ```json
    {
      "mcpServers": {
        "arxiv-search": {
          "type": "stdio",
          "command": "docker",
          "args": [
            "run",
            "-i",
            "--rm",
            "arxiv-searcher-mcp"
          ]
        }
      }
    }
    ```

4. **Start the server from VS Code**

---


### Option 2: Local configuration with uv (requires code locally)

1.  **Create `.vscode/mcp.json`:**
    In your project root, create the `.vscode` folder if it doesn't exist. Inside, create un archivo llamado `mcp.json`.

2.  **Agrega la configuración local del servidor:**
    Copia y pega la siguiente configuración en `.vscode/mcp.json`:

    ```json
    {
      "servers": {
        "arxiv-search": {
          "command": "uv",
          "args": [
            "run",
            "${workspaceFolder}/arxiv_searcher/arxiv_mcp.py"
          ]
        }
      }
    }
    ```

3. **Inicia el servidor desde VS Code**

---

## ✨ Features

### 🛠️ Available Tools

This MCP server exposes several useful tools for searching, analyzing, and exporting arXiv papers in the field of software engineering:

- **search_papers**: Searches arXiv papers filtered by the Software Engineering category (`cs.SE`).
  - Parameters: `query`, `max_results`, `start_date`, `end_date`, `sort_by_relevance`, `category`
  - Returns: Dictionary with the query used and the results.

- **get_paper_details**: Gets detailed information about a paper by its arXiv ID.
  - Parameters: `arxiv_id`
  - Returns: Title, authors, abstract, dates, categories, DOI, etc.

- **search_by_author**: Searches for papers by author, with optional category and date filters.
  - Parameters: `author_name`, `max_results`, `category`, `start_date`, `end_date`
  - Returns: List of found papers.

- **analyze_paper_trends**: Analyzes trends in a collection of papers (authors, keywords, timeline, categories).
  - Parameters: `papers`, `analysis_type`
  - Returns: Statistics and analysis according to the requested type.

- **find_related_papers**: Finds related papers based on the title of a reference paper, using keyword similarity.
  - Parameters: `paper_title`, `max_results`, `similarity_threshold`, `category`
  - Returns: List of similar papers.

- **download_paper_pdf**: Downloads the PDF of an arXiv paper.
  - Parameters: `pdf_url`, `save_path`, `filename`
  - Returns: Path and status of the download.

- **export_search_results**: Exports search results to various formats (`bibtex`, `csv`, `json`, `markdown`).
  - Parameters: `results`, `format`, `filename`, `save_path`
  - Returns: Path to the exported file and a preview of the content.

- **get_arxiv_categories**: Returns the list of arXiv categories and their descriptions.
  - Parameters: None
  - Returns: Dictionary of categories and usage notes.

---

## 🧑‍💻 Example Usage

Here's how you can call the tool from a compatible MCP client:

```bash
@arxiv-search.search_papers(query="secure software development lifecycle from 2022", max_results=5)
```

This will search for the 5 most relevant papers since 2022 in the software engineering category.

---

## 🛠️ Development

### 📦 Install dependencies

Set up your virtual environment and install the required packages:

```bash
uv sync
```

**Or using Docker:**

```bash
docker build -t arxiv-searcher-mcp .
```

### ▶️ Run for development

Start the server directly from your terminal:

```bash
uv run --directory arxiv_searcher/ arxiv_mcp.py
```

### 🐞 Debugging

For an interactive debugging experience, use [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
# Option 1: Using MCP Inspector
npx @modelcontextprotocol/inspector uv run --directory arxiv_searcher/arxiv_mcp.py

# Option 2: Using fastmcp CLI
fastmcp dev arxiv_searcher/arxiv_mcp.py
```

When launched, the Inspector will provide a URL to view and debug server communications in your browser. Don't forget to copy the session token!

---

## 👤 Author

Developed by [emi-dm](https://emi-dm.github.io/).

💡 Contributions and improvements are welcome! Feel free to open a Pull Request (PR) if you have suggestions or enhancements.

---

## 📚 License

This project is licensed under the MIT License.