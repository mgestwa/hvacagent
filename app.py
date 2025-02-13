import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Konfiguracja
class Config:
    PAGE_TITLE = "HVAC Analysis Assistant"
    PAGE_ICON = "🔧"
    LAYOUT = "wide"
    MODEL_TEMPERATURE = 0.7
    MODEL_MAX_TOKENS = 2000

# Szablony promptów
class Prompts:
    ANALYSIS_TEMPLATE = """Jesteś asystentem analizy HVAC. Twoim zadaniem jest analizowanie dokumentacji technicznej HVAC oraz wyciąganie najważniejszych danych technicznych w sposób uproszczony i zrozumiały.

    Przeanalizuj poniższą dokumentację i przedstaw wyniki w formie listy. Upewnij się, że uwzględnisz wszystkie istotne parametry, w tym:
    - Typ i przeznaczenie układu
    - Parametry techniczne centrali
    - Parametry wentylatorów
    - Parametry wymienników
    - Parametry filtrów
    - Parametry automatyki
    - Certyfikaty i zgodności

    Dokumentacja:
    {input}

    Wynik analizy (w formie listy z myślnikami):"""

    FULL_ANALYSIS_QUERY = """
    {doc_text}
    
    Proszę wyciągnąć i przedstawić wszystkie istotne dane techniczne w formie listy.
    """

    BRIEF_ANALYSIS_QUERY = """
    {doc_text}
    
    Proszę wyodrębnić tylko najważniejsze parametry techniczne w formie krótkiej listy.
    """

class HVACParametersInput(BaseModel):
    """Inputs for HVAC parameter analysis"""
    text: str = Field(description="Fragment dokumentacji HVAC do analizy")

class HVACAnalysisTool(BaseTool):
    name: str = "hvac_analysis"
    description: str = "Analizuje parametry techniczne instalacji HVAC"
    args_schema: Type[BaseModel] = HVACParametersInput

    def _run(self, text: str) -> str:
        return text

    async def _arun(self, text: str) -> str:
        raise NotImplementedError("Async not supported")

class HVACAnalyzer:
    def __init__(self):
        load_dotenv()
        self.setup_streamlit()
        self.initialize_session_state()
        self.setup_llm_chain()

    def setup_streamlit(self):
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT
        )

    def initialize_session_state(self):
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def setup_llm_chain(self):
        llm = OpenAI(
            temperature=Config.MODEL_TEMPERATURE,
            max_tokens=Config.MODEL_MAX_TOKENS,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = PromptTemplate(
            template=Prompts.ANALYSIS_TEMPLATE,
            input_variables=["input"]
        )
        
        self.chain = prompt | llm

    def create_sidebar(self):
        with st.sidebar:
            st.header("Ustawienia")
            analysis_type = st.selectbox(
                "Typ analizy",
                ["Pełna analiza", "Tylko parametry techniczne"]
            )
            
            st.header("O aplikacji")
            st.write("""
            Asystent wykorzystuje sztuczną inteligencję do analizy dokumentacji HVAC.
            Może pomóc w:
            - Analizie parametrów technicznych
            - Sprawdzaniu zgodności z normami
            - Sugerowaniu optymalizacji
            """)
            return analysis_type

    def create_main_interface(self):
        st.title("🔧 Asystent Analizy HVAC")
        st.write("Wprowadź dokumentację techniczną do analizy")
        
        doc_text = st.text_area(
            "Dokumentacja techniczna",
            height=200,
            placeholder="Wprowadź tekst dokumentacji do analizy..."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            analyze_button = st.button("Analizuj")
        with col2:
            clear_button = st.button("Wyczyść historię")
        with col3:
            export_button = st.button("Eksportuj raport")
            
        return doc_text, analyze_button, clear_button, export_button

    def analyze_document(self, doc_text: str, analysis_type: str):
        with st.spinner("Analizuję dokumentację..."):
            try:
                query_template = Prompts.FULL_ANALYSIS_QUERY if analysis_type == "Pełna analiza" else Prompts.BRIEF_ANALYSIS_QUERY
                analysis_query = query_template.format(doc_text=doc_text)
                
                result = self.chain.invoke({"input": analysis_query})
                
                st.success("Analiza zakończona!")
                st.write("### Wynik analizy:")
                st.write(result)
                
                self.update_history(doc_text, result)
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas analizy: {str(e)}")

    def update_history(self, doc_text: str, result: str):
        st.session_state.chat_history.append(("user", doc_text))
        st.session_state.chat_history.append(("assistant", result))

    def display_history(self):
        if st.session_state.chat_history:
            st.write("### Historia analiz:")
            for role, content in st.session_state.chat_history:
                if role == "user":
                    st.info(f"📝 Dokumentacja:\n{content}")
                else:
                    st.success(f"🔍 Analiza:\n{content}")

    def clear_history(self):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

    def export_report(self):
        if st.session_state.chat_history:
            report = "# Raport z analizy HVAC\n\n"
            for role, content in st.session_state.chat_history:
                report += f"## {'Dokumentacja' if role == 'user' else 'Analiza'}\n"
                report += f"{content}\n\n"
            
            st.download_button(
                label="Pobierz raport",
                data=report,
                file_name="raport_hvac.md",
                mime="text/markdown"
            )

    def run(self):
        analysis_type = self.create_sidebar()
        doc_text, analyze_button, clear_button, export_button = self.create_main_interface()
        
        if analyze_button and doc_text:
            self.analyze_document(doc_text, analysis_type)
        
        self.display_history()
        
        if clear_button:
            self.clear_history()
            
        if export_button:
            self.export_report()

if __name__ == "__main__":
    analyzer = HVACAnalyzer()
    analyzer.run()
    