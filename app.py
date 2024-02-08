import os
import lancedb
import asyncio
import warnings
import threading
from git import Repo
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from langchain.vectorstores import LanceDB
from langchain.text_splitter import Language
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferMemory


warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
db = lancedb.connect(os.environ["LANCEDB_PATH"])
llm = ChatOpenAI(
    model_name="gpt-4", openai_api_key=os.environ["OPENAI_API_KEY"])


def clone_repo(repo_url, repo_path):
    parsed_url = urlparse(repo_url)
    repo_name = os.path.splitext(os.path.basename(parsed_url.path))[0]

    repo_folder_path = os.path.join(repo_path, repo_name)

    if os.path.exists(repo_folder_path):

        if os.listdir(repo_folder_path):
            print(f"Repository folder at {repo_folder_path} is not empty")
            return repo_folder_path
        else:
            print(
                f"Repository folder already exists at {repo_folder_path}, but is empty")

    if not os.path.exists(repo_folder_path):
        os.makedirs(repo_folder_path)

    repo = Repo.clone_from(repo_url, to_path=repo_folder_path)
    return repo_folder_path


def convert_path_to_url(file_path):
    file_path = file_path.replace("\\", "/")

    return file_path


async def process_upload(repo_path, uuid, programming_language):
    language_extensions = {
        "Python": [".py", ".ipynb"],
        "Javascript": [".js"]
    }

    extensions = language_extensions[programming_language]
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=extensions,
        parser=LanguageParser(parser_threshold=500),
        show_progress=True,
    )

    documents = loader.load()

    splitter = ""
    if programming_language == "Javascript":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=int(os.environ["CHUNK_SIZE"]), chunk_overlap=200)

    elif programming_language == "Typescript":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.TS, chunk_size=int(os.environ["CHUNK_SIZE"]), chunk_overlap=200)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=int(os.environ["CHUNK_SIZE"]), chunk_overlap=200)

    texts = splitter.split_documents(documents)
    # embeddings = OpenAIEmbeddings(disallowed_special=(
    # ), openai_api_key=os.environ["OPENAI_API_KEY"])

    # db_name = uuid
    # # exist = db.open_table(db_name)

    # # if not exist:
    # table = db.create_table(db_name, data=[
    #     {"vector": embeddings.embed_query(
    #         "Hello World"), "text": "Hello World", "id": uuid}
    # ], mode="overwrite")
    # LanceDB.from_documents(texts, embeddings, connection=table)

    html_files = [os.path.join(root, file) for root, dirs, files in os.walk(
        repo_path) for file in files if file.endswith(".html")]

    if html_files:
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*",
            suffixes=[".html"],
            parser=LanguageParser(parser_threshold=500),
            show_progress=True,
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.HTML, chunk_size=int(os.environ["CHUNK_SIZE"]), chunk_overlap=200
        )

        texts2 = splitter.split_documents(documents)
        texts = texts + texts2

    db_name = uuid
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=os.environ["OPENAI_API_KEY"])

    table = db.create_table(db_name, data=[{"vector": embeddings.embed_query(
        "Hello World"), "text": "Hello World", "id": uuid}], mode="overwrite")

    LanceDB.from_documents(texts, embeddings, connection=table)
    print("Done")


def process_upload_thread(repo_path, uuid, programming_language):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_upload(
        repo_path, uuid, programming_language))
    loop.close()


@app.route('/load_repo', methods=['POST'])
def load_repo():
    try:
        data = request.get_json()
        programming_language = data['programming_language']
        uuid = data['uuid']
        repo_url = data["repo_url"]

        media_dir = os.path.join(os.getcwd(), 'media')
        if not os.path.exists(media_dir):
            os.mkdir(media_dir)

        repo_path = clone_repo(repo_url, media_dir)
        repo_path = convert_path_to_url(media_dir)

        if uuid != "11fef45d-d548-4c16-a671-d655a8ba3e97":
            threading.Thread(target=process_upload_thread, args=(
                repo_path, uuid, programming_language), daemon=True).start()

        return jsonify({'message': 'Repository Cloned Successfully.'}), 200
    except Exception as e:
        return jsonify({'message': 'Repository Failed to clone.'}), 400


@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        uuid = data['uuid']

        db_name = uuid
        table = db.open_table(db_name)
        vectorstore = LanceDB(table, OpenAIEmbeddings(disallowed_special=(
        ), openai_api_key=os.environ["OPENAI_API_KEY"]))

        conversation_memory = ConversationBufferMemory(
            memory_key="history", input_key="question")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={'k': 4, 'lambda_mult': 0.25, 'score_threshold': 0.8}),
            # verbose=True,
            chain_type_kwargs={
                # "verbose": True,
                "memory": conversation_memory,
            })

        result = qa(question)

        return jsonify({"answer": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gen_question', methods=['POST'])
def gen_question():

    data = request.get_json()
    language = data['programming_language']
    uuid = data['uuid']

    if uuid == "11fef45d-d548-4c16-a671-d655a8ba3e97":

        response = ["""Explain the purpose of the "ExportAllQuestions" class.""",
                    """What is the functionality of the "add_data_recursively" function?""",
                    """How does the "MainMenuView" route work?"""]
        return jsonify({"questions": response})

    db_name = uuid
    table = db.open_table(db_name)
    vectorstore = LanceDB(table, OpenAIEmbeddings(disallowed_special=(
    ), openai_api_key=os.environ["OPENAI_API_KEY"]))

    llm_gpt3 = ChatOpenAI(model_name="gpt-3.5-turbo",
                          openai_api_key=os.environ["OPENAI_API_KEY"])

    conversation_memory = ConversationBufferMemory(
        memory_key="history", input_key="question")

    qa = RetrievalQA.from_chain_type(
        llm=llm_gpt3,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={'k': 4, 'lambda_mult': 0.25, 'score_threshold': 0.8}),
        # verbose=True,
        chain_type_kwargs={
            # "verbose": True,
            "memory": conversation_memory,
        })

    if language == "Python":
        prompt = """
            // AI role
            List all api routes, functions,classes for attached document AI response must be only a valid json output file meant for data import. 
            Required: no AI commentary or tutorial, no AI introduction nor markdown. AI generated output will be directly written to a json file.
            
            
            //Output format
            {
            "api_routes":[ "api_route1", "api_route2", "api_route3"],
            "function":[ "function1", "function2", "function3"],
            "classes":[ "class1", "class2", "class3"]
            }
            """

        result = qa(prompt)
        result = result['result']

    else:
        prompt = """
            // AI role
            List all functions for attached document AI response must be only a valid json output file meant for data import. 
            Required: no AI commentary or tutorial, no AI introduction nor markdown. AI generated output will be directly written to a json file.
            
            
            //Output format
            {
            "function":[ "function1", "function2", "function3"]
            }
            """

        result = qa(prompt)
        result = result['result']
    # print(result)
    prompt = f"""
        // AI role
            Task: Generate 3 questions form the text shared below.

            //Language Specification: 
            specific programming language for which the default questions need to be generated 
            programming language is {language}
            
            
            //Instructions:
            {result}
            using this  list of functions or api_routes or classes, generate 3 questions. 
            Questions should ask to explain a function or explain the working of a route or API etc.
            If api_routes, functions and classes are available, generate 1 question from each
            Note that question should not contain programming language like "in Python" or "Javascript" etc.
            
            //Output format 
            AI response must be only a valid array of questions seperated by comma. 
            Required: no AI commentary or tutorial, no AI introduction nor markdown. AI generated output will be directly written to a output.
            
            //format of output (AVOID UNECESSARY CHARACTERS LIKE SLASHES. Each question seperated by comma ):
            question1, question2, question3
        """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate 3 questions form the text shared below as per the instructions."},
            {"role": "user", "content": prompt}
        ]
    )
    # db_name = uuid
    # table = db.open_table(db_name)
    # vectorstore = LanceDB(table, OpenAIEmbeddings(disallowed_special=(
    # ), openai_api_key=os.environ["OPENAI_API_KEY"]))

    # conversation_memory = ConversationBufferMemory(
    #     memory_key="history", input_key="question")
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(
    #         search_kwargs={'k': 4, 'lambda_mult': 0.25, 'score_threshold': 0.8}),
    #     # verbose=True,
    #     chain_type_kwargs={
    #         # "verbose": True,
    #         "memory": conversation_memory,
    #     })

    # result = qa(prompt)
    response = str(completion.choices[0].message.content)
    # print(response)
    response = response.split(",") if "," in response else response.split("\n")

    return jsonify({"questions": response})


@app.route('/')
def hello():
    return "hello"


if __name__ == '__main__':
    app.run(debug=True)
