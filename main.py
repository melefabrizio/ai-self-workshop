from langchain.vectorstores.qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.cohere import ChatCohere

from langchain_community.llms import Cohere
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import CohereEmbeddings

import boto3
import dotenv
from langchain_community.llms import Bedrock


def bedrock(model_name="anthropic.claude-v2:1", temperature=0.1):
    session = boto3.Session(profile_name="genai")
    client = session.client('bedrock-runtime', region_name="us-west-2")
    model = Bedrock(client=client,
                    model_id=model_name,
                    model_kwargs={"temperature": temperature, "max_tokens_to_sample": 1000})
    return model


def vector_store():
    loader = TextLoader("resources/formule.txt")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\r", "\t"],
        chunk_size=150,
        chunk_overlap=0
    )
    chunks = loader.load_and_split(text_splitter)
    vectorstore = Qdrant.from_documents(
        chunks,
        location=":memory:",
        collection_name="formule",
        embedding=CohereEmbeddings()
    )
    return vectorstore


def retrieve(query):
    vectorstore = vector_store()
    results = vectorstore.search(query, k=5, search_type='similarity')
    return results


def request_parser():
    llm = Cohere(model="command", max_tokens=256, temperature=0.1)
    template = open("resources/parser.txt").read(
    )  # read the template from file
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatCohere(llm=llm)
    chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser())

    return chain


def main():
    llm = Cohere(model="command", max_tokens=256, temperature=0.1)
    template = open("resources/expression_generator.txt").read(
    )  # read the template from file
    prompt = ChatPromptTemplate.from_template(template)
    model = bedrock(model_name="anthropic.claude-v2:1", temperature=0.1)
    request_chain = request_parser()

    chain = (
        prompt
        | model
        | StrOutputParser()
    )
    # ask user for input
    while True:
        query = input("Inserisci una richiesta: ")
        parsed_query = request_chain.invoke({"input": query})
        formulas = retrieve(query)
        print("Formule trovate: \n\n", formulas)
        print("Parsed query: \n\n", parsed_query)
        print("\n\n\n")
        print(chain.invoke({"input": parsed_query, "formula": formulas}))
        print("\n\n")



if __name__ == '__main__':
    dotenv.load_dotenv()
    main()
