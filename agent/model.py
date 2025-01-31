from langchain_openai import ChatOpenAI
MODEL = "deepseek-chat"
def load_model():
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    model = ChatOpenAI(name=MODEL, base_url="https://api.aitunnel.ru/v1/")
    return model
model = load_model()
