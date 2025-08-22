from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
# import camelot
from dotenv import load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
import requests
import trafilatura
# import pdfplumber
import requests, bs4
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

##artciles 

# url_1 = "http://today.lorientlejour.com/article/1461192/everything-keeps-getting-more-expensive-price-hikes-in-lebanon-continue.html"
# downloaded = trafilatura.fetch_url(url_1)
# article_1 = trafilatura.extract(downloaded)

url_2 = "https://www.anera.org/blog/the-cost-of-living-in-lebanon-in-2024"
downloaded = trafilatura.fetch_url(url_2)
article_2 = trafilatura.extract(downloaded)

url_3 = "https://reliefweb.int/report/lebanon/lebanon-economic-monitor-spring-2025-turning-tide-enar#:~:text=The%20report%20also%20includes%20special,structural%20constraints%20and%20widespread%20dollarization"
downloaded = trafilatura.fetch_url(url_3)
article_3 = trafilatura.extract(downloaded)

url_4 = "https://thisisbeirut.com.lb/articles/1318587/the-world-bank-unveils-a-roadmap-to-revitalize-the-economy-in-2025#:~:text=The%20World%20Bank%20also%20expects,reforms%20to%20ensure%20their%20sustainability"
downloaded = trafilatura.fetch_url(url_4)
article_4 = trafilatura.extract(downloaded)

url_5 = "https://www.ice.it/it/news/notizie-dal-mondo/287501"
downloaded = trafilatura.fetch_url(url_5)
article_5 = trafilatura.extract(downloaded)

# url_7 = "https://today.lorientlejour.com/article/1448783/inflation-hits-16-annually-in-january-monthly-rise-slows-to-11.html#:~:text=Miscellaneous%20goods%20and%20services"
# downloaded = trafilatura.fetch_url(url_7)
# article_7 = trafilatura.extract(downloaded)


##wiki page + beautiful soup

url_wiki = f"https://api.wikimedia.org/core/v1/wikipedia/en/page/Lebanese_liquidity_crisis/html"
html = requests.get(url_wiki, headers={"User-Agent":"your-app/1.0"}).text
soup = bs4.BeautifulSoup(html, "html.parser")
wiki_ariticle = "/n".join(p.get_text(" ", strip=True) for p in soup.select("section p"))

url_wiki_2 = f"https://en.wikipedia.org/api/rest_v1/page/html/Economy_of_Lebanon"
html = requests.get(url_wiki_2, headers={"User-Agent":"your-app/1.0"}).text
soup = bs4.BeautifulSoup(html, "html.parser")
wiki_ariticle_2 = "/n".join(p.get_text(" ", strip=True) for p in soup.select("section p"))

def make_desc(text: str, max_words=40):
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

docs = [
    Document(
        page_content=article_2,
        metadata={
            "source": "anera",
            "url": url_2,
            "type": "article",
            "description": make_desc(article_2), 
            "title": "Cost of Living in Lebanon"  
        },
    ),
    Document(
        page_content=article_3,
        metadata={
            "source": "reliefweb",
            "url": url_3,
            "type": "article",
            "description": make_desc(article_3),
            "title": "Lebanon Economic Monitor Spring 2025"
        },
    ),
    Document(
        page_content=article_4,
        metadata={
            "source": "thisisbeirut",
            "url": url_4,
            "type": "article",
            "description": make_desc(article_4),
            "title": "World Bank Roadmap 2025"
        },
    ),
    Document(
        page_content=article_5,
        metadata={
            "source": "ice.it",
            "url": url_5,
            "type": "article",
            "description": make_desc(article_5),
            "title": "ICE Italy note on Lebanon"
        },
    ),
    Document(
        page_content=wiki_ariticle,
        metadata={
            "source": "wikipedia",
            "url": url_wiki,
            "type": "wiki",
            "description": make_desc(wiki_ariticle),
            "title": "Lebanese liquidity crisis"
        },
    ),
    Document(
        page_content=wiki_ariticle_2,
        metadata={
            "source": "wikipedia",
            "url": url_wiki_2,
            "type": "wiki",
            "description": make_desc(wiki_ariticle_2),
            "title": "Economy of Lebanon"
        },
    ),
]


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# huggingface_embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )

spacy_chunker = SpacyTextSplitter(pipeline="en_core_web_sm",chunk_size=1000)
spacy_chunks = spacy_chunker.split_documents(docs)

article_vectorstore = FAISS.from_documents(spacy_chunks, embedding=embeddings)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(spacy_chunks)


###other attempts

query = "Does lebanon have inflation"
results = vectorstore.similarity_search(query, k=2)
print(results[0].page_content)



##RECURSIVE CHUNKER PRODUCED meh resulst
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,     
#     chunk_overlap=50,   
# )

# chunks = splitter.split_documents(docs)

#Semantic has better results
# semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
# semantic_chunks = semantic_chunker.split_documents(docs)


##plumber failed at capturing the tables in the pdfs

# with pdfplumber.open("april2025_en.pdf") as pdf:
#     page = pdf.pages[5]   # choose page
#     table = page.extract_table({
#         "vertical_strategy": "lines",
#         "horizontal_strategy": "text",
#         "snap_tolerance": 3
#     })
#     for row in table:
#         print(row)

########camelot works well with semi structured data like pdfs with tables/images

# gov_file_path = 'https://www.economy.gov.lb/media/14782/april2025_en.pdf'
# aub_file_path = "https://www.aub.edu.lb/oip/internationals/Documents/Cost-Of-Living-In-Beirut.pdf"


# gov_pdf_tables = camelot.read_pdf(gov_file_path, pages="all", flavor="stream")

# aub_pdf_tables = camelot.read_pdf(aub_file_path, pages="all", flavor="stream")


# gov_ppf_tables = gov_pdf_tables[2:] ####drop first two, camelot stream tends too put everything as a table
# aub_pdf_tables = aub_pdf_tables[2:] #####drop because it also picked up text as tables for the first two tables

# pdf = OnlinePDFLoader("https://www.aub.edu.lb/oip/internationals/Documents/Cost-Of-Living-In-Beirut.pdf")
