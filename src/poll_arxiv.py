import arxiv
from datetime import datetime, timedelta
from sync_github import update_github
import pickle, os, time

# arXiv Computer Science categories and their descriptions
categories:dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.RO": "Robotics",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "stat.ML": "Machine Learning (Statistics Category)"
}

def get_arxiv_papers(client, query: str, max_results: int = 5) -> list[arxiv.Result]:
    """
    Fetches papers from arXiv based on a query.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    results = []
    for result in client.results(search):
        results.append(result)
    return results


def get_arxiv_papers_content(end_date: datetime):
    """
    Fetches the top 10 papers from each category in Computer Science from arXiv.
    """
    # Create a client to interact with arXiv
    # The client is used to fetch papers from arXiv
    # The page size is set to 1000 to fetch more results in one go
    client = arxiv.Client(page_size=1000, num_retries=3, delay_seconds=5)
    start_date = end_date - timedelta(days=1)
    content = f"# Computer Science arXiv Papers\n\nCollection of top 10 Computer Science research papers pulled daily from arXiv.\n\n---\n\nPulled on {end_date} PST.\n\n"
    # Iterate through each category and fetch the papers
    for category, description in categories.items():
        # Construct the search query
        # The search query is a combination of the category and the date range
        # The date range is from the start_date to the end_date
        search_query = f"cat:{category} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
        # Fetch the papers
        papers = get_arxiv_papers(client, search_query, max_results=10)
        # Iterate through the papers and format the content
        # The content is formatted in Markdown
        cat_content = ""
        idx = 1
        for paper in papers:
            cat_content += f"### {idx}. {paper.title}\n\n"
            cat_content += f"[{paper.title}]({paper.pdf_url})\n\n"
            cat_content += f"Authors: {', '.join([author.name for author in paper.authors])}\n\n"
            cat_content += f"{paper.summary}\n\n"
            idx += 1
        # If there are papers in the category, add them to the content
        if len(cat_content) > 0:
            content += f"### {description}\n\n"
            content += cat_content
    return content

# Main function to run the script
end_date = datetime.now()
# Check if the script has been run before
# If it has, load the end_date from the pickle file
# If it hasn't, set the end_date to the current date minus one day
if os.path.exists("arxiv_papers.pkl"):
    with open("arxiv_papers.pkl", "rb") as f:
        end_date = pickle.load(f)
else:
    end_date = end_date - timedelta(days=1)
# task loop
# The loop runs indefinitely, checking if it is time to fetch new papers
while True:
    timenow = datetime.now()
    # If the current time is more than a day, fetch new papers
    if timenow - end_date > timedelta(days=1):
        end_date = timenow
        content = get_arxiv_papers_content(end_date)
        with open("arxiv_papers.pkl", "wb") as f:
            pickle.dump(end_date, f)
        update_github(content)
        with open(f"{end_date.strftime('%Y%m%d')}.md", "w") as f:
            f.write(content)
        print(f"Updated papers for {end_date.strftime('%Y%m%d')}")
    time.sleep(3600)  # Sleep for an hour
