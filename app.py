from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain.retrievers import ArxivRetriever
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv
import json

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    task="text-generation",
    timeout=300
    )

chat_model = ChatHuggingFace(llm=llm)
retriever = ArxivRetriever()
SUMMARY_TEMPLATE = """{doc} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary =  SUMMARY_PROMPT | chat_model | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
)| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()

search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user", 
            """Write 2 arxiv search queries to search online that form an objective opinion from the following question:{question}
            you must respond with a string containing a list of strings in the following format:
            '["query 1", "query 2"]'
            (DO NOT format the output using any programming language, and output only one list in string queries and respond with plain text)
            """
        )
    ]
)

search_question_chain = search_prompt | chat_model | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 800 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            WRITER_SYSTEM_PROMPT
        ), 
        (
            "user", 
            RESEARCH_REPORT_TEMPLATE
        )
    ]
)

# helper function to group all list strings together
def collapse_list_of_lists(list_of_lists):
    content = list()
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | chat_model | StrOutputParser()


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-assistant",
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
