import os
import uuid
from dotenv import load_dotenv

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required

from .models import ChatSession, ChatMessage
from .forms import SignUpForm, CustomLoginForm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain_community.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()


FAISS_INDEX_PATH = "scpwikiapp/faiss_index.index"

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH)
    return None


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username,password=password)
            login(request,user)
    else:
        form = SignUpForm()
    return render()

def login(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.post)
        if form.is_valid():
            username = form.cleaned_data.get(username)
            password = form.cleaned_data.get(password)
            user = authenticate(username=username,password=password)
            if user is not None:
                login(request,user)
                return()
    else:
        form = CustomLoginForm()
    return()

#this is for starting new chat. it is linked with the function below it by foreign key
def start_chat_session(request):
    session_id = str(uuid.uuid4())
    ChatSession.objects.create(session_id=session_id,user=request.user.username)
    return redirect('chat', session_id=session_id)

#make this happen every time they click the button to get their question answered and saves it to the chat session above
def handle_chat(request,session_id):
    chat_session = get_object_or_404(ChatSession,session_id=session_id)

    if request.method == 'POST':
        user_message = request.POST.get('message','')
        if user_message:
            usermessage = ChatMessage.objects.create(session=chat_session,sender='user',message=user_message)
            response = ragapp(user_message)
            botmessage = ChatMessage.objects.create(session=chat_session,sender='bot', message=response)
            return redirect('chat',session_id=session_id)
    return redirect('chat',session_id=session_id)

def ragapp(question):
    # Load the FAISS index if it exists, otherwise create a new one
    faiss_index = load_faiss_index()
    if not faiss_index:
        loader = PDFLoader(file_path="scpwikiapp/ragpdfs/jailbird.pdf")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        openai_api_key = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(FAISS_INDEX_PATH)

    retriever = faiss_index.as_retriever()

    info_prompt_template = PromptTemplate.from_template("""
    You are an expert on the game SCP: Secret Laboratory. Answer the following question based on the game:

    Question: {question}
    Answer:
    """)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0)

    memory = ConversationBufferMemory()
    tools = [
        Tool(
            name="InformationProvider",
            func=lambda query: retriever.get_relevant_documents(query),
            description="Use this tool to retrieve detailed information about SCP: Secret Laboratory."
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=50
    )

    response = agent(question)
    return response

