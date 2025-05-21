

import cohere
co = cohere.Client("7vh0EuMhDhYc454iO8bQRQyUy51XVmsNMkPRoqQt")

def optimize_query_with_cohere(raw_query):
    prompt = f"""You are a smart multilingual product search assistant. 
Given this query: "{raw_query}", detect its language, translate to English, fix typos, and rewrite it as a clean product search query."""
    
    response = co.generate(prompt=prompt, model="command-r", max_tokens=50)
    return response.generations[0].text.strip()
