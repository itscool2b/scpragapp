import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, MapReduceDocumentsChain

load_dotenv()



urls = ['https://en.scpslgame.com/index.php?title=SCP-049',
'https://en.scpslgame.com/index.php?title=SCP-049-2',
'https://en.scpslgame.com/index.php?title=SCP-079',
'https://en.scpslgame.com/index.php?title=SCP-096',
'https://en.scpslgame.com/index.php?title=SCP-106',
'https://en.scpslgame.com/index.php?title=SCP-173',
'https://en.scpslgame.com/index.php?title=SCP-939',
'https://en.scpslgame.com/index.php?title=SCP-018',
'https://en.scpslgame.com/index.php?title=SCP-207',
#'https://en.scpslgame.com/index.php?title=SCP-207%3F',
'https://en.scpslgame.com/index.php?title=SCP-244',
'https://en.scpslgame.com/index.php?title=SCP-268',
'https://en.scpslgame.com/index.php?title=SCP-500',
'https://en.scpslgame.com/index.php?title=SCP-1576',
'https://en.scpslgame.com/index.php?title=SCP-1853',
'https://en.scpslgame.com/index.php?title=SCP-2176',
'https://en.scpslgame.com/index.php?title=SCP-330',
'https://en.scpslgame.com/index.php?title=SCP-914',
'https://en.scpslgame.com/index.php?title=SCP-559',
'https://en.scpslgame.com/index.php?title=SCP-956',
'https://en.scpslgame.com/index.php?title=SCP-1507',
'https://en.scpslgame.com/index.php?title=SCP-2536_(Disambiguation)',
'https://en.scpslgame.com/index.php?title=Class-D_Personnel',
'https://en.scpslgame.com/index.php?title=Scientist',
'https://en.scpslgame.com/index.php?title=Chaos_Insurgent',
'https://en.scpslgame.com/index.php?title=Facility_Guard',
'https://en.scpslgame.com/index.php?title=Mobile_Task_Force',
'https://en.scpslgame.com/index.php?title=Spectator',
'https://en.scpslgame.com/index.php?title=SCP-3114',
'https://en.scpslgame.com/index.php?title=Tutorial',
'https://en.scpslgame.com/index.php?title=Filmmaker',
'https://en.scpslgame.com/index.php?title=Spectator#Overwatch',
'https://en.scpslgame.com/index.php?title=COM-15',
'https://en.scpslgame.com/index.php?title=COM-18',
'https://en.scpslgame.com/index.php?title=Crossvec',
'https://en.scpslgame.com/index.php?title=FR-MG-0',
'https://en.scpslgame.com/index.php?title=FSP-9',
'https://en.scpslgame.com/index.php?title=MTF-E11-SR',
'https://en.scpslgame.com/index.php?title=.44_Revolver',
'https://en.scpslgame.com/index.php?title=AK',
'https://en.scpslgame.com/index.php?title=Logicer',
'https://en.scpslgame.com/index.php?title=Shotgun',
'https://en.scpslgame.com/index.php?title=Flashbang_Grenade',
'https://en.scpslgame.com/index.php?title=High-Explosive_Grenade',
'https://en.scpslgame.com/index.php?title=3-X_Particle_Disruptor',
'https://en.scpslgame.com/index.php?title=A7',
'https://en.scpslgame.com/index.php?title=COM-45',
'https://en.scpslgame.com/index.php?title=Jailbird',
'https://en.scpslgame.com/index.php?title=Micro_H.I.D.',
'https://en.scpslgame.com/index.php?title=Ammunition',
'https://en.scpslgame.com/index.php?title=Armor_(Disambiguation)',
'https://en.scpslgame.com/index.php?title=Bag_of_Candies',
'https://en.scpslgame.com/index.php?title=Coin',
'https://en.scpslgame.com/index.php?title=Flashlight',
'https://en.scpslgame.com/index.php?title=Keycard',
'https://en.scpslgame.com/index.php?title=Radio',
'https://en.scpslgame.com/index.php?title=Adrenaline',
'https://en.scpslgame.com/index.php?title=First_Aid_Kit',
'https://en.scpslgame.com/index.php?title=Painkillers',
'https://en.scpslgame.com/index.php?title=Lantern',
'https://en.scpslgame.com/index.php?title=Coal',
'https://en.scpslgame.com/index.php?title=Tape_Player',
'https://en.scpslgame.com/index.php?title=Snowball',
'https://en.scpslgame.com/index.php?title=Light_Containment_Zone',
'https://en.scpslgame.com/index.php?title=Heavy_Containment_Zone',
'https://en.scpslgame.com/index.php?title=Entrance_Zone',
'https://en.scpslgame.com/index.php?title=Surface_Zone',
#'https://en.scpslgame.com/index.php?title=Bulletproof_Locker_%E2%84%967',
'https://en.scpslgame.com/index.php?title=Emergency_Power_Stations',
'https://en.scpslgame.com/index.php?title=First_Aid_Cabinet',
'https://en.scpslgame.com/index.php?title=MTF-E11-SR_Rack',
'https://en.scpslgame.com/index.php?title=Standard_Locker',
'https://en.scpslgame.com/index.php?title=Weapon_Locker_Type_21',
'https://en.scpslgame.com/index.php?title=Workstation',
'https://en.scpslgame.com/index.php?title=C.A.S.S.I.E.',
'https://en.scpslgame.com/index.php?title=Alpha_Warhead',
'https://en.scpslgame.com/index.php?title=Clutter_System',
'https://en.scpslgame.com/index.php?title=Decontamination_Process',
'https://en.scpslgame.com/index.php?title=Inventory_Mechanics',
'https://en.scpslgame.com/index.php?title=Overcharge',
'https://en.scpslgame.com/index.php?title=Secondary_HP_(Disambiguation)',
'https://en.scpslgame.com/index.php?title=Spawning_Mechanics',
'https://en.scpslgame.com/index.php?title=Stamina',
'https://en.scpslgame.com/index.php?title=Status_Effects',
'https://en.scpslgame.com/index.php?title=Victory_Conditions',
'https://en.scpslgame.com/index.php?title=Achievements',
'https://en.scpslgame.com/index.php?title=Soundtrack',
]




loader = WebBaseLoader(urls)
docs = loader.load()

textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = textsplitter.split_documents(docs)

openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=openai_api_key, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vector = FAISS.from_documents(documents,embeddings)
retriever = vector.as_retriever()
vector.save_local("faiss_index.index")

info_prompt_template = PromptTemplate.from_template("""
You are an expert on the game SCP: Secret Laboratory or also named Scp SL. Answer the following question based on the game:

Question: {question}
Answer:
""")
map_prompt_template = PromptTemplate.from_template("""
Create detailed and challenging questions about many components of the game SCP: Secret Laboratory. These are the many docs you can refer to:

"{docs}"
Questions:
""")


reduce_prompt_template = PromptTemplate.from_template("""
The following is a set of questions about components of the game SCP: Secret Laboratory as a test:
{docs}
Take these and distill them into final, consolidated questions from any components.
Questions:
""")



def generate_questions(docs):
   
    map_chain = LLMChain(llm=llm, prompt=map_prompt_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt_template)
    
    
    map_reduce = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_chain,
        document_variable_name='docs',
        return_intermediate_steps=False
    )
    return map_reduce.run(docs)

memory = ConversationBufferMemory()
print(all[:5])

def retrieve_documents(query):
    return retriever.get_relevant_documents(query)

tools = [
    Tool(
    name="InformationProvider",
    func=retrieve_documents,
    description="Use this tool to retrieve detailed information about SCP: Secret Laboratory."
),
    Tool(
        name="QuestionGenerator",
        func=generate_questions,
        description="Use this tool to generate questions from documents."
    )

]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=50
)
print("Agent: How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Agent: Goodbye!")
        break
    response = agent.run(user_input)
    print(f"Agent: {response}")