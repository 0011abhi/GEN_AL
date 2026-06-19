import getpass
from langchain import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from google.colab import auth, drive

auth.authenticate_user()
drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/GenerativeAI.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_content = file.read()

api_key = getpass.getpass("Enter your Cohere API Key: ")
llm = ChatCohere(cohere_api_key=api_key, model="command-a-03-2025")

template = """
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:
{text}

* Summary: Provide a concise summary of the document.
* Key Takeaways: List 3 important points from the text.
* Sentiment Analysis: Determine if the sentiment is Positive, Negative, or Neutral.
"""

prompt = PromptTemplate(input_variables=["text"], template=template)
formatted_prompt = prompt.format(text=text_content)

response = llm.invoke([HumanMessage(content=formatted_prompt)]).content
print(response)
