# 🚀 ArxivSearcher MCP Server

An MCP server for intelligently searching Software Engineering papers on arXiv, with advanced filtering and sorting.

---

## 📋 Prerequisites

Before you begin, make sure you have installed:
- [Python](https://www.python.org/downloads/) (3.11 or higher)
- [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver)
- [Node.js and npm](https://nodejs.org/en/download/) (for debugging with the MCP Inspector)

---

## ⚡️ Quickstart in VS Code

Follow these steps to get the server running in your workspace:

1.  **Create `.vscode/mcp.json`:**
    In your project root, create the `.vscode` folder if it doesn't exist. Inside, create a file named `mcp.json`.

2.  **Add the server configuration:**
    Copy and paste the following configuration into `.vscode/mcp.json` so VS Code knows how to run the server.

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

3. **Start the server**

---

## ✨ Features

### 🛠️ Tools Provided

This MCP server exposes several useful tools for searching, analyzing, and exporting arXiv papers in the field of software engineering:

#### `search_papers`
Searches arXiv papers filtered by the Software Engineering category (`cs.SE`).
- **Parameters:** `query`, `max_results`, `start_date`, `end_date`, `sort_by_relevance`, `category`
- **Returns:** Dictionary with the query used and the results.

#### `get_paper_details`
Gets detailed information about a paper by its arXiv ID.
- **Parameters:** `arxiv_id`
- **Returns:** Title, authors, abstract, dates, categories, DOI, etc.

#### `search_by_author`
Searches for papers by a specific author, with optional category and date filters.
- **Parameters:** `author_name`, `max_results`, `category`, `start_date`, `end_date`
- **Returns:** List of found papers.

#### `analyze_paper_trends`
Analyzes trends in a collection of papers (authors, keywords, timeline, categories).
- **Parameters:** `papers`, `analysis_type`
- **Returns:** Statistics and analysis according to the requested type.

#### `find_related_papers`
Finds related papers based on the title of a reference paper, using keyword similarity.
- **Parameters:** `paper_title`, `max_results`, `similarity_threshold`, `category`
- **Returns:** List of similar papers.

#### `download_paper_pdf`
Downloads the PDF of an arXiv paper.
- **Parameters:** `pdf_url`, `save_path`, `filename`
- **Returns:** Path and status of the download.

#### `export_search_results`
Exports search results to various formats (`bibtex`, `csv`, `json`, `markdown`).
- **Parameters:** `results`, `format`, `filename`, `save_path`
- **Returns:** Path to the exported file and a preview of the content.

#### `get_arxiv_categories`
Returns the list of arXiv categories and their descriptions.
- **Parameters:** None
- **Returns:** Dictionary of categories and usage notes.

---

## 🧑‍💻 Example Usage

Here's how you can call the tool from a compatible MCP client:

```
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

### ▶️ Run for development

Start the server directly from your terminal:

```bash
uv run --directory src/arxivsearcher/ arxiv_mcp.py
```

### 🐞 Debugging

For an interactive debugging experience, use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

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