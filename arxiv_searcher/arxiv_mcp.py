import arxiv
import re
import json
import os
import requests
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from fastmcp import FastMCP
import logging
import fitz  # PyMuPDF

mcp = FastMCP("Arxiv Searcher 🚀")


# MCP Prompts and Descriptions

@mcp.prompt
def search_by_author(author_name: str) -> str:
    """Generates a prompt to search for papers by an author."""
    return f"Search for the latest papers by '{author_name}' on Arxiv."


@mcp.prompt
def search_by_recent_topic(topic: str) -> str:
    """Generates a prompt to search for recent papers on a topic."""
    return f"Show me the most relevant papers on '{topic}' from the last year on Arxiv."

@mcp.prompt
def search_by_keyword_in_title(keyword: str) -> str:
    """Generates a prompt to search for a specific keyword in paper titles."""
    return f"Find papers with the keyword '{keyword}' directly in the title."

@mcp.prompt
def search_author_and_topic(author_name: str, topic: str) -> str:
    """Generates a combined prompt to search for papers by an author on a specific topic."""
    return f"Look for papers by '{author_name}' on the topic of '{topic}'."

@mcp.prompt
def search_in_category(topic: str, category: str) -> str:
    """Generates a prompt to search for a topic within a specific arXiv category."""
    return f"Search for '{topic}' within the arXiv category '{category}'."

@mcp.prompt
def get_latest_papers_in_category(category: str = "cs.LG") -> str:
    """Generates a prompt to get the most recent submissions in a category."""
    return f"What are the 5 newest papers in the '{category}' category?"

@mcp.prompt
def get_paper_by_id(arxiv_id: str) -> str:
    """Generates a prompt to fetch a specific paper by its ID."""
    return f"Get me the details for the paper with ArXiv ID: '{arxiv_id}'."


# Resources

@mcp.resource("data://arxiv_categories")
def get_arxiv_categories() -> str:
    """Provides the list of Arxiv categories in JSON format."""
    with open("arxiv_categories.json", "r") as f:
        categories = json.load(f)
    return categories


@mcp.resource("data://search_tips")
def get_search_tips() -> str:
    """Returns a document with tips for searching on Arxiv."""
    return """
    **Advanced ArXiv Search Guide for LLMs**

    This guide details how to build effective search queries for the ArXiv API.

    **1. Logical Operators:**
    - `AND`: Finds documents containing ALL terms.
      - Example: `electron AND positron`
    - `OR`: Finds documents containing ANY of the terms.
      - Example: `GAN OR "Generative Adversarial Network"`
    - `ANDNOT`: Excludes documents containing the term. (Less common, use with care).
      - Example: `transformer ANDNOT "electrical"`

    **2. Specific Field Searching:**
    You can limit your search to specific fields of a paper using prefixes.
    - `ti:` (Title): Searches only in the paper's title.
      - Example: `ti:"Quantum Computing"`
    - `au:` (Author): Searches by author name. For exact names, use quotes.
      - Example: `au:"Yann LeCun"`
    - `abs:` (Abstract): Searches only in the abstract.
      - Example: `abs:"self-attention mechanism"`
    - `cat:` (Category): Filters by a specific ArXiv category.
      - Example: `cat:cs.AI`

    **3. Combining Searches (Advanced Examples):**
    - **Author and Title:** Find papers by a specific author that contain a keyword in the title.
      - `au:"Yoshua Bengio" AND ti:consciousness`
    - **Topic in Multiple Categories:** Search for a topic across several categories of interest.
      - `(ti:"language model" OR abs:"language model") AND (cat:cs.CL OR cat:cs.LG)`
    - **Exclude a Secondary Topic:** Search for 'transformers' in computer vision, but exclude those that mention NLP.
      - `(abs:"computer vision" AND ti:transformer) ANDNOT abs:NLP`

    **4. Date Syntax:**
    - The `search_papers` tool handles dates with the `start_date` and `end_date` parameters.
    - It is preferable to use these parameters instead of including dates directly in the `query` for greater precision.
    """


@mcp.resource("data://mcp_readme")
def get_mcp_readme() -> str:
    """Provides the project's README.md file."""
    with open("README.md", "r") as file:
        return file.read()


@mcp.resource("data://downloaded_papers")
def list_downloaded_papers() -> str:
    """
    Reads all downloaded PDF files, extracts their content, and returns it
    in a structured XML-like format: <Paper {title}> {content} </Paper>.
    Requires the 'PyMuPDF' library to be installed (`pip install PyMuPDF`).
    """
    directory = "downloaded_papers"
    if not os.path.exists(directory):
        return ""  # Return empty string if no directory

    papers_content = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

    for filename in pdf_files:
        try:
            # 1. Extract arXiv ID (with version) from filename
            arxiv_id = filename.replace(".pdf", "")

            # 2. Fetch paper metadata (title) from arXiv API
            search = arxiv.Search(id_list=[arxiv_id])
            paper_meta = next(search.results(), None)
            title = paper_meta.title if paper_meta else f"Unknown Title for {arxiv_id}"

            # 3. Read PDF content
            filepath = os.path.join(directory, filename)
            content = ""
            with fitz.open(filepath) as doc:
                content = "".join(page.get_text() for page in doc)
            
            # 4. Format the output for this paper
            papers_content.append(f"<Paper {title} > {content} </Paper>")

        except Exception as e:
            logging.error(f"Failed to process file {filename}: {e}")
            continue
            
    return "\n\n".join(papers_content)


# MCP Tools


@mcp.tool
def search_papers(
    query: str,
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    sort_by_relevance: bool = True,
    category: str = "cs.SE",
) -> dict:
    """
    Search for papers on arXiv.
    It can parse natural language queries, extracting keywords and years for filtering.

    :param query: The base search query. Can be natural language.
    :param max_results: The maximum number of results to return.
    :param start_date: The start date for the search period (YYYY-MM-DD or YYYY). Overrides years in query.
    :param end_date: The end date for the search period (YYYY-MM-DD or YYYY). Overrides years in query.
    :param sort_by_relevance: If True, sorts by relevance. If False, sorts by submission date.
    :param category: The arXiv category to search in (e.g., 'cs.AI', 'cs.CL', 'cs.SE').
    """
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "the",
        "of",
        "in",
        "for",
        "to",
        "with",
        "on",
        "is",
        "are",
        "was",
        "were",
        "it",
    }

    # Extract years from query to use as date filters if not provided explicitly
    years_in_query = re.findall(r"\b(20\d{2})\b", query)
    query_text = re.sub(r"\b(20\d{2})\b", "", query).strip()

    # Use provided dates or fall back to dates from query
    effective_start_date = start_date
    if not effective_start_date and years_in_query:
        effective_start_date = min(years_in_query)

    effective_end_date = end_date
    if not effective_end_date and years_in_query:
        effective_end_date = max(years_in_query)

    # Process keywords from the query text
    keywords = [
        word
        for word in query_text.split()
        if word.lower() not in STOP_WORDS and len(word) > 2
    ]

    if keywords:
        # Build a structured query from keywords, joining with OR for broader results
        keyword_query = " OR ".join([f'(ti:"{kw}" OR abs:"{kw}")' for kw in keywords])
        query_parts = [f"({keyword_query})"]
    else:
        # Fallback to using the original query text if no keywords are left
        query_parts = [f'(ti:"{query_text}" OR abs:"{query_text}")']

    if category:
        query_parts.append(f"cat:{category}")

    # Add date range to the query
    if effective_start_date or effective_end_date:
        start = "19910814"
        if effective_start_date:
            try:
                dt = datetime.strptime(effective_start_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.strptime(effective_start_date, "%Y")
            start = dt.strftime("%Y%m%d")

        end = datetime.now().strftime("%Y%m%d")
        if effective_end_date:
            try:
                dt = datetime.strptime(effective_end_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.strptime(effective_end_date, "%Y")
                dt = dt.replace(month=12, day=31)
            end = dt.strftime("%Y%m%d")

        query_parts.append(f"submittedDate:[{start} TO {end}]")

    final_query = " AND ".join(query_parts)
    print(f"[arxiv-search] Query sent: {final_query}")

    sort_criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by_relevance
        else arxiv.SortCriterion.SubmittedDate
    )

    search = arxiv.Search(
        query=final_query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for r in search.results():
        results.append(
            {
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "summary": r.summary,
                "pdf_url": r.pdf_url,
                "published_date": r.published.strftime("%Y-%m-%d"),
            }
        )
    return {"query_used": final_query, "results": results}


@mcp.tool
def get_paper_details(arxiv_id: str) -> dict:
    """
    Get detailed information about a specific paper by ArXiv ID.

    :param arxiv_id: The ArXiv ID (e.g., '2301.12345')
    """
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        return {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published_date": paper.published.strftime("%Y-%m-%d"),
            "updated_date": paper.updated.strftime("%Y-%m-%d"),
            "categories": paper.categories,
            "primary_category": paper.primary_category,
            "arxiv_id": paper.entry_id.split("/")[-1],
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
        }
    except Exception as e:
        return {"error": f"Failed to fetch paper details: {str(e)}"}


@mcp.tool
def search_by_author(
    author_name: str,
    max_results: int = 20,
    category: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Search papers by a specific author.

    :param author_name: Name of the author to search for
    :param max_results: Maximum number of results
    :param category: Optional category filter (e.g., 'cs.SE', 'cs.AI')
    :param start_date: Optional start date filter (YYYY-MM-DD or YYYY)
    :param end_date: Optional end date filter (YYYY-MM-DD or YYYY)
    """
    query_parts = [f'au:"{author_name}"']

    if category:
        query_parts.append(f"cat:{category}")

    # Add date range if specified
    if start_date or end_date:
        start = "19910814"
        if start_date:
            try:
                dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.strptime(start_date, "%Y")
            start = dt.strftime("%Y%m%d")

        end = datetime.now().strftime("%Y%m%d")
        if end_date:
            try:
                dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                dt = datetime.strptime(end_date, "%Y")
                dt = dt.replace(month=12, day=31)
            end = dt.strftime("%Y%m%d")

        query_parts.append(f"submittedDate:[{start} TO {end}]")

    final_query = " AND ".join(query_parts)
    print(f"[arxiv-search] Author query: {final_query}")

    search = arxiv.Search(
        query=final_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for r in search.results():
        results.append(
            {
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "summary": r.summary,
                "pdf_url": r.pdf_url,
                "published_date": r.published.strftime("%Y-%m-%d"),
                "arxiv_id": r.entry_id.split("/")[-1],
                "categories": r.categories,
            }
        )

    return {
        "author": author_name,
        "query_used": final_query,
        "total_results": len(results),
        "results": results,
    }


@mcp.tool
def analyze_paper_trends(
    papers: List[Dict[str, Any]], analysis_type: str = "authors"
) -> dict:
    """
    Analyze trends in a collection of papers.

    :param papers: List of papers from search_papers results
    :param analysis_type: Type of analysis ('authors', 'keywords', 'timeline', 'categories')
    """
    if not papers or "results" not in papers:
        if isinstance(papers, list):
            results = papers
        else:
            return {
                "error": "Invalid papers format. Expected list or dict with 'results' key."
            }
    else:
        results = papers["results"]

    if not results:
        return {"error": "No papers to analyze"}

    analysis = {}

    if analysis_type == "authors":
        author_counts = Counter()
        for paper in results:
            for author in paper.get("authors", []):
                author_counts[author] += 1

        analysis = {
            "type": "authors",
            "total_unique_authors": len(author_counts),
            "most_prolific_authors": author_counts.most_common(10),
            "collaboration_stats": {
                "avg_authors_per_paper": sum(len(p.get("authors", [])) for p in results)
                / len(results),
                "single_author_papers": sum(
                    1 for p in results if len(p.get("authors", [])) == 1
                ),
                "multi_author_papers": sum(
                    1 for p in results if len(p.get("authors", [])) > 1
                ),
            },
        }

    elif analysis_type == "timeline":
        date_counts = Counter()
        for paper in results:
            date = paper.get("published_date", "")
            if date:
                year = date.split("-")[0]
                date_counts[year] += 1

        analysis = {
            "type": "timeline",
            "papers_by_year": dict(sorted(date_counts.items())),
            "most_active_year": date_counts.most_common(1)[0] if date_counts else None,
            "total_years_span": len(date_counts),
        }

    elif analysis_type == "categories":
        category_counts = Counter()
        for paper in results:
            categories = paper.get("categories", [])
            for cat in categories:
                category_counts[cat] += 1

        analysis = {
            "type": "categories",
            "total_categories": len(category_counts),
            "most_common_categories": category_counts.most_common(10),
            "category_distribution": dict(category_counts),
        }

    elif analysis_type == "keywords":
        # Extract keywords from titles and abstracts
        text_content = []
        for paper in results:
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            text_content.append(f"{title} {summary}")

        if text_content:
            try:
                # Use TF-IDF to find important terms
                vectorizer = TfidfVectorizer(
                    max_features=50, stop_words="english", ngram_range=(1, 2), min_df=2
                )
                tfidf_matrix = vectorizer.fit_transform(text_content)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1

                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)

                analysis = {
                    "type": "keywords",
                    "top_keywords": keyword_scores[:20],
                    "total_unique_terms": len(feature_names),
                }
            except Exception as e:
                analysis = {
                    "type": "keywords",
                    "error": f"Could not perform keyword analysis: {str(e)}",
                    "fallback_word_count": Counter(),
                }

    analysis["total_papers_analyzed"] = len(results)
    return analysis


@mcp.tool
def find_related_papers(
    paper_title: str,
    max_results: int = 10,
    similarity_threshold: float = 0.7,
    category: str | None = None,
) -> dict:
    """
    Find papers related to a given paper title using keyword similarity.

    :param paper_title: Title of the reference paper
    :param max_results: Maximum number of related papers to return
    :param similarity_threshold: Minimum similarity score (0.0 to 1.0)
    :param category: Optional category filter
    """
    try:
        # Extract keywords from the title
        stop_words = {
            "a",
            "an",
            "and",
            "the",
            "of",
            "in",
            "for",
            "to",
            "with",
            "on",
            "is",
            "are",
            "was",
            "were",
            "it",
        }

        keywords = [
            word.lower()
            for word in re.findall(r"\b\w+\b", paper_title)
            if word.lower() not in stop_words and len(word) > 2
        ]

        if not keywords:
            return {"error": "No meaningful keywords found in title"}

        # Create search query from keywords
        keyword_query = " OR ".join([f'(ti:"{kw}" OR abs:"{kw}")' for kw in keywords])
        query_parts = [f"({keyword_query})"]

        if category:
            query_parts.append(f"cat:{category}")

        final_query = " AND ".join(query_parts)

        # Search for related papers
        search = arxiv.Search(
            query=final_query,
            max_results=max_results * 2,  # Get more results to filter by similarity
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []

        for r in search.results():
            # Calculate simple similarity based on keyword overlap
            paper_text = f"{r.title} {r.summary}".lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in paper_text)
            similarity = matches / len(keywords) if keywords else 0

            if similarity >= similarity_threshold:
                results.append(
                    {
                        "title": r.title,
                        "authors": [a.name for a in r.authors],
                        "summary": r.summary[:500] + "..."
                        if len(r.summary) > 500
                        else r.summary,
                        "pdf_url": r.pdf_url,
                        "published_date": r.published.strftime("%Y-%m-%d"),
                        "similarity_score": round(similarity, 3),
                        "arxiv_id": r.entry_id.split("/")[-1],
                    }
                )

        # Sort by similarity score and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:max_results]

        return {
            "reference_title": paper_title,
            "keywords_used": keywords,
            "similarity_threshold": similarity_threshold,
            "total_related_found": len(results),
            "related_papers": results,
        }

    except Exception as e:
        return {"error": f"Failed to find related papers: {str(e)}"}


@mcp.tool
def export_search_results(
    results: Dict[str, Any],
    format: str = "bibtex",
    filename: str | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Export search results to various formats.

    :param results: Results from search_papers or other search functions
    :param format: Export format ('bibtex', 'csv', 'json', 'markdown')
    :param filename: Output filename (without extension)
    :param save_path: Directory to save the file (default: current directory)
    """
    try:
        if save_path is None:
            save_path = os.getcwd()

        os.makedirs(save_path, exist_ok=True)

        # Extract papers from results
        if isinstance(results, dict) and "results" in results:
            papers = results["results"]
        elif isinstance(results, list):
            papers = results
        else:
            return {
                "error": "Invalid results format. Expected a list of papers or a dict with a 'results' key."
            }

        if not papers:
            return {"error": "No papers to export."}

        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_search_{timestamp}"

        full_path = os.path.join(save_path, f"{filename}.{format}")

        if format == "bibtex":
            bibtex_entries = []
            query_info = results.get("query_used", "N/A")
            export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            header = f"""% Query: {query_info}
% Exported: {export_time}
"""
            bibtex_entries.append(header)

            bibtex_keys = set()
            for i, paper in enumerate(papers):
                authors = paper.get("authors", ["unknown"])
                year = paper.get("published_date", "unknown").split("-")[0]

                first_author_lastname = "unknown"
                if authors and isinstance(authors, list) and authors[0] != "unknown":
                    name_parts = authors[0].split(" ")
                    if name_parts:
                        first_author_lastname = name_parts[-1]

                first_author_lastname = re.sub(
                    r"[^a-zA-Z0-9]", "", first_author_lastname
                ).lower()

                key = f"{first_author_lastname}{year}"

                # Handle duplicates
                original_key = key
                suffix = 1
                while key in bibtex_keys:
                    key = f"{original_key}_{suffix}"
                    suffix += 1
                bibtex_keys.add(key)

                title = paper.get("title", "No Title Provided")
                author_str = " and ".join(paper.get("authors", []))
                pdf_url = paper.get("pdf_url", "")

                arxiv_id_match = (
                    re.search(r"/pdf/([^v]+)", pdf_url) if pdf_url else None
                )
                if arxiv_id_match:
                    arxiv_id = arxiv_id_match.group(1)
                    journal = f"arXiv preprint arXiv:{arxiv_id}"
                else:
                    journal = f"arXiv preprint arXiv:{key}"

                entry = f"""@article{{{key},
    title = {{{title}}},
    author = {{{author_str}}},
    year = {{{year}}},
    journal = {{{journal}}},
    url = {{{pdf_url}}}
}}"""
                bibtex_entries.append(entry)

            content = "\n\n".join(bibtex_entries)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        elif format == "csv":
            df = pd.DataFrame(papers)
            df.to_csv(full_path, index=False, encoding="utf-8-sig")
            content = df.to_string()

        elif format == "json":
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=4)
            content = json.dumps(papers, indent=4)

        elif format == "markdown":
            md_entries = []
            for paper in papers:
                title = paper.get("title", "N/A")
                authors = ", ".join(paper.get("authors", ["N/A"]))
                date = paper.get("published_date", "N/A")
                url = paper.get("pdf_url", "#")
                summary = paper.get("summary", "N/A").replace("\n", " ")

                md_entries.append(
                    f"""### {title}\n**Authors:** {authors}\n**Published:** {date}\n**[PDF Link]({url})**\n> {summary}\n"""
                )

            content = "\n---\n".join(md_entries)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        else:
            return {"error": f"Unsupported format: {format}"}

        return {
            "success": True,
            "format": format,
            "saved_path": full_path,
            "papers_exported": len(papers),
            "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to export results: {str(e)}"}

@mcp.tool
def download_paper(arxiv_id: str, directory: str = "downloaded_papers") -> dict:
    """
    Downloads the PDF of a paper to a local directory on the server.
    NOTE: In a stateless/free hosting environment, this file is temporary and
    will be deleted when the server restarts or sleeps.

    :param arxiv_id: The ArXiv ID of the paper to download (e.g., '2301.12345').
    :param directory: The local directory where the paper will be saved.
    """
    try:
        # Ensure the download directory exists
        os.makedirs(directory, exist_ok=True)
        
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Define a clean filename to avoid issues with special characters
        clean_id = re.sub(r'[^0-9v.]', '_', arxiv_id)
        filename = f"{clean_id}.pdf"
        
        # Download the paper to the specified directory
        paper.download_pdf(dirpath=directory, filename=filename)
        
        filepath = os.path.join(directory, filename)
        logging.info(f"Paper {arxiv_id} downloaded to {filepath}")
        
        return {
            "success": True,
            "arxiv_id": arxiv_id,
            "local_path": filepath,
            "message": f"Paper is temporarily available at the server path: {filepath}"
        }
        
    except StopIteration:
        logging.error(f"Paper with ID {arxiv_id} not found.")
        return {"success": False, "error": f"Paper with ID {arxiv_id} not found."}
    except Exception as e:
        logging.error(f"Failed to download paper {arxiv_id}: {e}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
