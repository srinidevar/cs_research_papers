from github import Github

def update_github(content): 
    token = "<YOUR_GITHUB_TOKEN>"
    g = Github(token)
    repo = g.get_repo("srinidevar/cs_research_papers")
    file = repo.get_contents("README.md")
    repo.update_file(file.path, "Updating README.md", content, file.sha)
    g.close()