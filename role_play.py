# !/usr/bin/python3
"""
# -*-coding:utf-8-*-
# Author: Phoss.Xu
# Email: phosssuki@gmail.com
# CreateDate:  
# Description:
"""
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask
from flask import request


# Load, chunk and index the contents of the blog.
def get_chain():
    loader = TextLoader("data/text/microsoft_news_api.txt")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    all_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_type="similarity")

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

def load_question():
    result = []
    with open('data/text/question.txt') as file:
        result = file.readlines()
    return [question.strip('\n') for question in result]

def write_answer(data: dict| None):
    with open('data/text/answer.txt', 'a') as file:
        for index, (question, answer) in enumerate(data.items(), start=1):
            file.writelines(f'Q{index}. {question}\nA{index}:{answer}\n\n')

def run():
    chain = get_chain()
    question_list = load_question()
    data = {}
    for question in question_list:
        answer = chain.invoke(f'{question},给出答案后，换行给出依据序号。')
        data[question] = answer
    write_answer(data)

app = Flask(__name__)


@app.route('/question')
def question():
    params = request.args
    question = params.get('question')
    answer  = chain.invoke(question)
    return answer


if __name__ == '__main__':
    chain = get_chain()
    app.run()

