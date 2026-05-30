from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
def main():

    # testing the llm
    llm = ChatOllama(model="gemma4")
    response = llm.invoke("What is the capital of France?")
    print(response.content)

if __name__ == "__main__":
    main()
