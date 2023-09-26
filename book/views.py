from django.shortcuts import render,redirect
from book.models import Book,Message
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from review.settings import MEDIA_ROOT,MEDI
from django.core.files.storage import FileSystemStorage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pickle
import openai
from django.http import JsonResponse
from pathlib import Path
import os
from decouple import config

# Create your views here.
# here we take in the text and convert it to chunks
def chunk_data(text):
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    # length_function=len
    )
    chunks = text_splitter.split_text(text)
    # docs = text_splitter.split_documents(docs)
    return chunks
# here we convert pdf_to text
def convert(pdf):
    text = ''
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
# convert text_chunk to get_vectorstore
def get_vectorstore(text_chunks,pdf):
    openai.api_key=config('api_key')
    embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embedding)
    with open(pdf,'wb') as f:
            pickle.dump(vectorstore,f)
    return vectorstore

def conversations(vectorstore):
    prompt_template = """you are a helpful assistant that help review books.
    {context}
    question:{question}
    Answer here:"""

    PROMPT =PromptTemplate(
    template=prompt_template,input_variables=["context","question"])
    openai.api_key = config('api_key')
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True,output_keys='answer')
    conversation= ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003",temperature=0.7,openai_api_key=openai.api_key),
          memory=memory,retriever=vectorstore.as_retriever(),combine_docs_chain_kwargs={'prompt':PROMPT}
)

    return(conversation)

def home(request):
    books = Book.objects.values('pdf')
    names=Book.objects.all()
    base= str(MEDIA_ROOT)
    print(base)
    
    return render (request,"base/menu.html",{'book':names})





def upload(request):
    # uploading of pdf file``
    if request.method == 'POST':
        pdf= request.FILES['pdf']
        name = request.POST.get('title') 
        fs = FileSystemStorage()
        pkl= fs.path(name+'.pkl')
        pk2=fs.url(name+'.pkl')
    
        text=convert(pdf)
        chunk=chunk_data(text)
        vector=get_vectorstore(chunk,pkl)
        
        # saving of pickle file in database
        book = Book.objects.create(
        reader = request.user,
        name = name,
        pdf =  pk2,
        )
        
        return redirect('home')
      
    return render (request,"base/upload.html",)

def conversation(request,pk):
    vector = Book.objects.get(pk=pk)
    user=request.user
    messages=Message.objects.filter(user=user)
    message_list=list(reversed(messages))
    vectors=str(vector.pdf)

    base= str(MEDI)
    vectorpath= str(base + vectors)
    
    with open(vectorpath,'rb') as f:
        vectorstore = pickle.load(f)
        
    qa=conversations(vectorstore)
    chat=None
    chats=[]
    if request.method == 'POST':
        body=request.POST.get('body')
        chat=qa({'question':body})

        
        for i,message in enumerate(chat['chat_history']):
        
            if i%2 ==0:
                q=message.content
                question={'question':q}

            else:
                a=message.content
                answer={'answer':a}
                d={**question,**answer}
        messages=  Message.objects.create(
            user = request.user,
            ai =d['answer'],
            human=d['question']
        )
        return redirect('review',pk=vector.id)

    return render (request,"base/chat.html",{'chat':message_list})


  