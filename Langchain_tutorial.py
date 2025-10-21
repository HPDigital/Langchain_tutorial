"""
Langchain_tutorial
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
load_dotenv()


# In[3]:


from langchain_openai import ChatOpenAI


# ### INSTANCIANDO EL MODELO

# In[5]:


llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.7, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


# ### INVOKE

# In[9]:


response = llm.invoke("Hola, como estas?") # invoke solo acepta una pregunta
print(response.content)


# ### BATCH

# In[5]:


response = llm.batch(["Hola, como estas?", "Escribe un poema sobre de Bolivia"])
print(response[0].content)  # batch acepta varias preguntas a la vez y las corre en paralelo
print("**************************")
print(response[1].content)


# ### STREAM

# In[6]:


response = llm.stream("Escribe un poema sobre de Bolivia") # Steam responde en pedasos (chunks) separados con lo que pidamos
for chunk in response:
    print(chunk.content,end="", flush = True)


# ### TEMPLATE

# In[7]:


# ejemplo antes de usar el template
response = llm.invoke("Cuenta una broma sobre gallinas") # con temaplte haremos que gallina sea una replazada por una variable
print(response.content)


# #### TEMPLATE USANDO .FROM_TEMAPLATE()

# In[8]:


# ejemplo 1 usando temaplate
from langchain_core.prompts import ChatPromptTemplate
# Aqui creamos el template
prompt = ChatPromptTemplate.from_template("Cuenta una broma sobre {sujeto}")
# ahora creamos la cadena ("LLM chain")
chain = prompt | llm # hacemos que el pormpt sea pasado a la instacia LLM

# pasamos el promot o instruccion usando el temaplate
response = chain.invoke({"sujeto":"perros"}) #ojo cambiamos LLM por CHAIN y con el diccionario remplazamos gallina por perros
print(response.content)



# #### TEMPLATE USANDO .FROM_MESSAGES()

# In[9]:


# ejemplo 2 usando temaplate
from langchain_core.prompts import ChatPromptTemplate
# Aqui creamos el template
prompt = ChatPromptTemplate.from_messages([
    ("system","Eres un cocinero de renombre, crea una receta con el siguiente ingrediente principal"),
    ("human", "{ingrediente}")
])
# ahora creamos la cadena ("LLM chain")
chain = prompt | llm # hacemos que el pormpt sea pasado a la instacia LLM

# pasamos el promot o instruccion usando el temaplate
response = chain.invoke({"ingrediente":"durazno"})
print(response.content)


# In[10]:


# ejemplo 3 usando temaplate
from langchain_core.prompts import ChatPromptTemplate
# Aqui creamos el template
prompt = ChatPromptTemplate.from_messages([
    ("system","Crea una lista de tres sinoninos de la siguiente palabra. El resultado debe ser palabras separadas con coma CSV"),
    ("human", "{ingrediente}")
])
# ahora creamos la cadena ("LLM chain")
chain = prompt | llm # hacemos que el pormpt sea pasado a la instacia LLM

# pasamos el promot o instruccion usando el temaplate
response = chain.invoke({"ingrediente":"durazno"})
print(response.content) # vemos que el resulatdo es string y no una lsita pythonera


# ### OUTPUT PARSER

# In[11]:


# ejemplo 4 usando temaplate y usandoo StrOutputParser para convertir el typo STR
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","Crea una lista de tres sinoninos de la siguiente palabra. El resultado debe ser palabras separadas con coma CSV"),
        ("human", "{ingrediente}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({"ingrediente":"durazno"})


print(call_string_output_parser()) # vemos que el resulatdo es string y no una lsita pythonera


# In[12]:


# ejemplo 5 usando temaplate y usando CommaSeparatedListOutputParser para convertir en LIST el resultado
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser



def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","Crea una lista de tres sinonimos de la siguiente palabra. El resultado debe ser palabras separadas con coma"),
        ("human", "{ingrediente}")
    ])

    parser = CommaSeparatedListOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({"ingrediente":"durazno"})


print(call_list_output_parser())


# In[13]:


# ejemplo 5 usando temaplate y usando CommaSeparatedListOutputParser para convertir en LIST el resultado
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","Crea una lista de tres sinonimos de la siguiente palabra.Indicando para cada sinonimo el pais donde mas se utiliza ese sinonimo \nFormating Instructions:{format_instructions}"),
        ("human", "{ingrediente}")
    ])

    class origen(BaseModel):
        sinonimo: str = Field(description = "La palabra sinonimo")
        pais: str = Field(description = "El pais donde mas se usa la palbra sinonimo")

    parser = JsonOutputParser(pydantin_object = origen)
    chain = prompt | llm | parser

    return chain.invoke({"ingrediente":"durazno", "format_instructions": parser.get_format_instructions()})


print(call_json_output_parser())


# # Conectando con fuentes externas

# In[14]:


#ejemplo 1 poniendo el contexto manulmente

from langchain_core.prompts import ChatPromptTemplate
llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
Contexto: LangChain Expression Language, o LCEL, es una forma declarativa de encadenar componentes LangChain. 
LCEL se diseñó desde el día 1 para permitir la puesta en producción de prototipos, sin cambios de código, 
desde la cadena más simple de “mensaje + LLM” hasta las cadenas más complejas 
(hemos visto a personas ejecutar con éxito cadenas LCEL con cientos de pasos en producción).

Pregunta: {input}
""")

chain = prompt | llm

response = chain.invoke({
    "input":"Que es LCEL"

})

print(response.content)


# In[15]:


# ejemplo 2 usando "import document" con un solo documento docA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 

docA = Document(
    page_content = """LangChain Expression Language, o LCEL, es una forma declarativa de encadenar componentes LangChain. 
LCEL se diseñó desde el día 1 para permitir la puesta en producción de prototipos, sin cambios de código, 
desde la cadena más simple de “mensaje + LLM” hasta las cadenas más complejas 
(hemos visto a personas ejecutar con éxito cadenas LCEL con cientos de pasos en producción). """

)

llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context}
                                            Pregunta: {input}
                                            """)

chain = prompt | llm

response = chain.invoke({
                            "input":"Que es LCEL",
                            "context" :[docA]
                            })

print(response.content)


# In[16]:


# ejemplo 3 usando "import document" con varios documentos y usando create_stuf_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from langchain.chains.combine_documents import create_stuff_documents_chain



docA = Document(
    page_content = """LangChain Expression Language, o LCEL, es una forma declarativa de encadenar componentes LangChain. 
LCEL se diseñó desde el día 1 para permitir la puesta en producción de prototipos, sin cambios de código, 
desde la cadena más simple de “mensaje + LLM” hasta las cadenas más complejas 
(hemos visto a personas ejecutar con éxito cadenas LCEL con cientos de pasos en producción). """

)

llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context} 
                                            Pregunta: {input}
                                            """)

#context es mejor que se escriba en ingles
# chain = prompt | llm 

chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

response = chain.invoke({ "input":"Que es LCEL","context" :[docA]})

print(response)


# ### EXTRAYENDO INFORMACION PARA CONTEXTO DE UNA PAGINA EXTERNA-->  ESTA ES LA PARTE LOAD

# In[17]:


# esto sirbve para cargar una pagina entera de informacion (pero ojo no estamos chunkenisando y sale caro)
# ejemplo 4 usando "import document" con varios documentos y usando create_stuf_documents_chain y webbsaseloader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load() 
    print(type(docs))
    return docs

docs = get_documents_from_web("https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language")

llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context} 
                                            Pregunta: {input}
                                            """)

#context es mejor que se escriba en ingles
# chain = prompt | llm 

chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

response = chain.invoke({ "input":"Que es LCEL","context" :docs})

print(response)


# In[18]:


type(docs)


# ### CHUNKENISANDO TEXTOS GANDES --> ESTA ES LA PARTE TRANSFROM

# In[19]:


#ahora si chunkenisarermos con text splitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load() 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

docs = get_documents_from_web("https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language")

llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context} 
                                            Pregunta: {input}
                                            """)

#context es mejor que se escriba en ingles
# chain = prompt | llm 

chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

response = chain.invoke({ "input":"Que es LCEL","context" :docs})

print(response)


# ### ESTA ES LA PARTE EMBEED (VECTORIZANDO) Y CARGANDO A UNA BASE DE DATOS CREADA EN LOCAL

# In[20]:


# en el ejemplo pasado tengo 263 documentitos, y lo que quiero es NO pasarlos todos sino solo los mas relevantes
#para eso se vectorizara cada chunk y se cargara de foamr separada con OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load() 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs



def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embeddings)
    return vectorStore

docs = get_documents_from_web("https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language")
vectorStore = create_db(docs)


llm = ChatOpenAI(model = "gpt-4o",
                 temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                 verbose = True                
                )


prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                            Contexto: {context} 
                                            Pregunta: {input}
                                            """)

#context es mejor que se escriba en ingles
# chain = prompt | llm 

chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

response = chain.invoke({ "input":"Que es LCEL","context" :docs})

print(response)


# ### PARTE RETRIVAL

# In[21]:


#ya se cargó los vectores y se creo una basew da ddatos d eventores ahora toca poder encontrar los mas relevantes a mi pregunta
# Lo que hace es carga todo los documentos, los separa en chuca, despeus cada chuk es vectorizado 
# y se gunarad en la base de datos, despues buca los chunks mas similares a la pregunta y esos los pasa como contexto
# de sta forma se ahorra en el costo de tokens y no se sobrepasa los limites

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load() 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs



def create_db(docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embeddings)
    return vectorStore


def create_chain(vectorStore):
    llm = ChatOpenAI(model = "gpt-4o",
                     temperature = 0.4, # mas cerca de 0 mas concreto mas cerca de 1 mas creativo
                     verbose = True                
                    )

    prompt = ChatPromptTemplate.from_template(""" Responde a las preguntas del ususario:
                                                Contexto: {context} 
                                                Pregunta: {input}
                                                """)
    chain = create_stuff_documents_chain(llm = llm, prompt = prompt)

    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,chain )
    return retrieval_chain


docs = get_documents_from_web("https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({ "input":"Que es LCEL"})# "context" :docs --> sacamos

print(response["answer"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






if __name__ == "__main__":
    pass
