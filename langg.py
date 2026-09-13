from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

model = ChatOpenAI()

prompt_1 = PromptTemplate.from_template(
    "Please summarize the following text: {note}",
    input_variables=["note"],
)
parser_1 = StrOutputParser()
mooel_2 = ChatAnthropic()
prompt_2 = PromptTemplate.from_template(
    "Please extract the key points from the following text: {note}",
    input_variables=["note"],
)   

parser_2 = StrOutputParser()
parallel_chain = RunnableParallel({
      'summarization': prompt_1 | model | parser_1,
        'key_points': prompt_2 | mooel_2 | parser_1
})
prompt_3 = PromptTemplate.from_template(
    "Please provide a final summary based on the following information:\n\nSummary: {summarization}\n\nKey Points: {key_points}",
    input_variables=["summarization", "key_points"],
)
parser_3 = StrOutputParser()
merged_chain = parallel_chain | prompt_3 | parser_3

result = merged_chain.invoke({"note": "LangChain is a framework for developing applications powered by language models. It provides tools and abstractions to simplify the process of building complex applications that leverage the capabilities of large language models."})
print(result)