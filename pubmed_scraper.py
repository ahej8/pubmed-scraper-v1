import requests
from bs4 import BeautifulSoup
import re
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def get_author_publications(author_name):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    query = f"{author_name}[Author]"
    url = f"{base_url}?term={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results_count = soup.find('span', class_='value')
    if results_count:
        return int(results_count.text.replace(',', ''))
    return 0

def generate_query(protein_target=''):
    related_terms = [
        "antibody", "immunoassay", "ELISA", "flow cytometry",
        "western blot", "immunohistochemistry", "neutralization",
        "immunoprecipitation", "therapeutic", "vaccine", "diagnostic"
    ]
    techniques = [
        "protein expression", "protein localization", "signaling pathway",
        "protein-protein interaction", "gene regulation", "knockout",
        "cellular function", "disease association", "biomarker",
        "drug development", "molecular mechanism"
    ]
    
    query_parts = []
    
    if protein_target:
        query_parts.append(f'("{protein_target}"[Title/Abstract] OR "{protein_target} protein"[Title/Abstract])')
    
    query_parts.extend([
        '(',
        ' OR '.join(f'"{term}"[Title/Abstract]' for term in related_terms),
        ')',
        'AND',
        '(',
        ' OR '.join(f'"{technique}"[Title/Abstract]' for technique in techniques),
        ')'
    ])
    
    return ' '.join(query_parts)

def scrape_pubmed(query, protein_target, max_results=50, max_publications=None):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    page = 1
    total_processed = 0
    
    while total_processed < max_results:
        url = f"{base_url}?term={query}&page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        articles = soup.find_all('article', class_='full-docsum')
        
        if not articles:
            break
        
        for article in articles:
            if total_processed >= max_results:
                break
            
            title_tag = article.find('a', class_='docsum-title')
            if title_tag:
                title = title_tag.text.strip()
                article_url = base_url + title_tag['href'].lstrip('/')
                
                authors_tag = article.find('span', class_='docsum-authors full-authors')
                authors = authors_tag.text.strip() if authors_tag else "No authors listed"
                
                first_author = authors.split(',')[0].strip() if authors != "No authors listed" else ""
                pub_count = get_author_publications(first_author) if first_author else 0
                
                if max_publications is not None and pub_count > max_publications:
                    continue
                
                abstract_text, summary = extract_abstract_and_summarize(article_url, protein_target)
                
                emails = extract_emails(article_url)
                
                if emails:
                    email_str = ', '.join(emails)
                else:
                    email_str = "No valid email found"

                yield {
                    'title': title,
                    'authors': authors,
                    'first_author_publications': pub_count,
                    'email': email_str,
                    'summary': summary,
                    'source_link': article_url
                }
                print(f"Processed article: {title}")
            
            total_processed += 1
        
        page += 1
        time.sleep(1)

def extract_emails(article_url):
    try:
        session = requests.Session()
        response = session.get(article_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b[A-Za-z0-9._%+-]+\s*\[at\]\s*[A-Za-z0-9.-]+\s*\[dot\]\s*[A-Z|a-z]{2,}\b',
            r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b'
        ]
        
        emails = set()
        
        # Check specific sections with higher priority
        priority_sections = ['author-list', 'affiliations', 'corresp-id', 'email', 'author-information']
        for section in priority_sections:
            section_tags = soup.find_all(['div', 'span', 'p', 'a'], class_=section)
            for section_tag in section_tags:
                for pattern in email_patterns:
                    found_emails = re.findall(pattern, section_tag.text, re.IGNORECASE)
                    emails.update(found_emails)
        
        # Check for mailto links
        mailto_links = soup.find_all('a', href=re.compile(r'^mailto:'))
        for link in mailto_links:
            emails.add(link['href'].replace('mailto:', ''))
        
        # If no emails found, check the entire page content
        if not emails:
            for pattern in email_patterns:
                emails.update(re.findall(pattern, response.text, re.IGNORECASE))
        
        # If still no emails found, try to follow "Author information" links
        if not emails:
            author_links = soup.find_all('a', text=re.compile(r'Author information', re.IGNORECASE))
            for link in author_links:
                author_url = "https://pubmed.ncbi.nlm.nih.gov" + link['href']
                author_response = session.get(author_url)
                author_soup = BeautifulSoup(author_response.text, 'html.parser')
                for pattern in email_patterns:
                    emails.update(re.findall(pattern, author_response.text, re.IGNORECASE))
        
        # Clean up and standardize emails
        cleaned_emails = set()
        for email in emails:
            cleaned = email.replace('[at]', '@').replace('[dot]', '.').replace(' ', '')
            if not re.match(r'^(example@|.*@example\.com)$', cleaned, re.IGNORECASE):
                cleaned_emails.add(cleaned)
        
        return list(cleaned_emails)
    except Exception as e:
        print(f"Error extracting emails from {article_url}: {e}")
        return []

def extract_abstract_and_summarize(article_url, protein_target):
    try:
        response = requests.get(article_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        abstract = soup.find('div', class_='abstract-content selected')
        if abstract:
            abstract_text = abstract.text.strip()
            summary = summarize_antibody_need(abstract_text, protein_target)
            return abstract_text, summary
        
        return "", "No abstract available."
    except Exception as e:
        print(f"Error extracting abstract from {article_url}: {e}")
        return "", "Error extracting abstract information."

def summarize_antibody_need(abstract_text, protein_target=''):
    # Define key terms and their associated reasons
    key_terms = {
        "immunoassay": "conducting sensitive protein detection assays",
        "ELISA": "performing enzyme-linked immunosorbent assays for quantitative analysis",
        "flow cytometry": "analyzing and sorting specific cell populations",
        "western blot": "detecting and quantifying specific proteins in complex mixtures",
        "immunohistochemistry": "visualizing protein distribution in tissue samples",
        "neutralization": "studying virus or toxin inhibition",
        "immunoprecipitation": "isolating and purifying specific proteins from cell lysates",
        "therapeutic": "developing targeted therapies or diagnostics",
        "vaccine": "formulating or evaluating vaccine candidates",
        "diagnostic": "creating diagnostic tests or kits",
        "antibody": "general antibody-related research"
    }
    
    if protein_target:
        key_terms[protein_target.lower()] = f"specific research on {protein_target}"
    
    # Tokenize the abstract into sentences
    sentences = sent_tokenize(abstract_text)
    
    # Find sentences containing key terms
    relevant_sentences = []
    for sentence in sentences:
        if any(term in sentence.lower() for term in key_terms):
            relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        return "No relevant antibody need found."
    
    # Use TF-IDF to find the most relevant sentence
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(relevant_sentences)
    
    # Calculate similarity to a query combining all key terms
    query = f"antibody {protein_target} " + " ".join(key_terms.keys())
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    most_relevant_idx = similarities.argmax()
    
    most_relevant_sentence = relevant_sentences[most_relevant_idx]
    
    # Identify the key terms in the most relevant sentence
    found_terms = [term for term in key_terms if term in most_relevant_sentence.lower()]
    
    if found_terms:
        reasons = [key_terms[term] for term in found_terms]
        if protein_target:
            return f"Potential need for antibodies for {', '.join(reasons)}, specifically related to {protein_target}."
        else:
            return f"Potential need for antibodies for {', '.join(reasons)}."
    else:
        if protein_target:
            return f"Specific antibody application for {protein_target} not clearly identified, but likely needed for general research purposes."
        else:
            return "Specific antibody application not clearly identified, but likely needed for general research purposes."