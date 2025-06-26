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
                  "--directory",
                  "${workspaceFolder}/arxiv_searcher/",
                  "run",
                  "arxiv_mcp.py"
              ]
          }
      }
    }
    ```

3. **Start the server**

---

## ✨ Features

### 🛠️ Tool: `search_papers`

This is the main tool provided by the server. It searches for papers on arXiv, always filtered for the Software Engineering category (`cs.SE`), and intelligently parses natural language queries.

**Parameters:**

- `query` (str): The search query. Can be a phrase like "multi-agent systems from 2023". Years and keywords are automatically extracted.
- `max_results` (int, optional): Maximum number of results to return. Defaults to `10`.
- `start_date` (str, opcional): Start date (YYYY or YYYY-MM-DD). Overrides any year found in the query.
- `end_date` (str, opcional): End date (YYYY or YYYY-MM-DD). Overrides any year found in the query.
- `sort_by_relevance` (bool, opcional): Sorts by relevance if `True`, otherwise by date. Defaults to `True`.

**Returns:**

A dictionary with `query_used` (the query sent to arXiv) and `results` (list of found papers).

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