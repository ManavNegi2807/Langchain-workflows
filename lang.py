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


prompt_template = PromptTemplate( 
               
    input_variables=["input_text"],
    template="Please summarize the following text: {input_text}",
   
)
output_parser=StrOutputParser();

chain = prompt_template| model |output_parser

input_text = "LangChain is a framework for developing applications powered by language models. It provides tools and abstractions to simplify the process of building complex applications that leverage the capabilities of large language models."

result = chain.invoke({"input_text": input_text})

print(result)

