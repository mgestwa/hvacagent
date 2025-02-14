import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
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
    MODEL_MAX_TOKENS = 1000  # Zmniejszamy maksymalną długość odpowiedzi
    CHUNK_SIZE = 1500  # Maksymalny rozmiar fragmentu tekstu do analizy
    CHUNK_OVERLAP = 200  # Nakładanie się fragmentów dla zachowania kontekstu


class HVACParametersInput(BaseModel):
    """Inputs for HVAC parameter analysis"""
    text: str = Field(description="Fragment dokumentacji HVAC do analizy")

class HVACTechnicalAnalysisTool(BaseTool):
    name: str = "hvac_technical_analysis"
    description: str = "Wyodrębnia parametry techniczne z dokumentacji HVAC"
    args_schema: Type[BaseModel] = HVACParametersInput

    def _run(self, text: str) -> str:
        template = """Wyodrębnij parametry techniczne wszystkich urządzeń HVAC z dokumentacji.

        INSTRUKCJE:
        1. Zidentyfikuj wszystkie urządzenia/centrale w tekście
        2. Dla każdego urządzenia wyodrębnij jego parametry
        3. Grupuj parametry według urządzeń
        4. Zachowaj wszystkie wartości liczbowe i jednostki

        FORMAT ODPOWIEDZI:
        [Nazwa/ID Urządzenia 1]
        - parametr: wartość (jednostka)
        - parametr: wartość (jednostka)

        [Nazwa/ID Urządzenia 2]
        - parametr: wartość (jednostka)
        - parametr: wartość (jednostka)

        PRZYKŁAD:
        [Centrala wentylacyjna NW1]
        - wydajność: 2000 m³/h
        - spręż: 300 Pa

        [Centrala wentylacyjna NW2]
        - wydajność: 1500 m³/h
        - spręż: 250 Pa

        TEKST DO ANALIZY:
        {text}"""
        
        return template.format(text=text)

    async def _arun(self, text: str) -> str:
        raise NotImplementedError("Async not supported")

class HVACComplianceTool(BaseTool):
    name: str = "hvac_compliance_check"
    description: str = "Sprawdza zgodność instalacji HVAC z normami i standardami"
    args_schema: Type[BaseModel] = HVACParametersInput

    def _run(self, text: str) -> str:
        template = """Sprawdź zgodność poniższej instalacji HVAC z normami:
        - Zgodność z PN-EN 13779
        - Zgodność z PN-EN 16798
        - Wymagania bezpieczeństwa
        - Certyfikaty i atesty
        
        Tekst do analizy:
        {text}
        """
        return template.format(text=text)

    async def _arun(self, text: str) -> str:
        raise NotImplementedError("Async not supported")

class HVACOptimizationTool(BaseTool):
    name: str = "hvac_optimization"
    description: str = "Sugeruje optymalizacje dla systemu HVAC"
    args_schema: Type[BaseModel] = HVACParametersInput

    def _run(self, text: str) -> str:
        template = """Jesteś ekspertem w dziedzinie systemów HVAC z głęboką wiedzą na temat efektywności energetycznej i strategii optymalizacji.

            <aktywacja_roli>
            Działasz jako wyspecjalizowany analityk systemów HVAC, skupiający się wyłącznie na technicznej analizie dokumentacji i generowaniu praktycznych rekomendacji.
            </aktywacja_roli>

            <cel_główny>
            Przeprowadzić szczegółową analizę dokumentacji systemu HVAC i wygenerować konkretne, wykonalne zalecenia optymalizacyjne, koncentrując się na efektywności energetycznej, możliwościach modernizacji, redukcji kosztów operacyjnych i poprawie wydajności.
            </cel_główny>

            <zasady_analizy>
            - ZAWSZE opieraj wszystkie rekomendacje na konkretnych danych z dostarczonej dokumentacji
            - BEZWZGLĘDNIE wyrażaj usprawnienia w mierzalnych wartościach (%, jednostki, koszty)
            - MUSISZ uwzględniać współzależności systemowe i potencjalne kompromisy
            - KATEGORYCZNIE przestrzegaj zgodności z normami HVAC
            - POD ŻADNYM POZOREM nie zalecaj modyfikacji mogących zagrozić bezpieczeństwu
            - W przypadku braku danych, ZAWSZE oznaczaj to jako "BRAK DANYCH"
            </zasady_analizy>

            <format_wyjściowy>
            Strukturyzuj odpowiedź dokładnie w następujących sekcjach:

            1. Efektywność Energetyczna:
            - Obecne wskaźniki efektywności
            - Konkretne możliwości usprawnień
            - Spodziewane korzyści (ilościowo)

            2. Potencjał Modernizacyjny:
            - Komponenty wymagające aktualizacji
            - Rekomendacje technologiczne
            - Złożoność wdrożenia (Niska/Średnia/Wysoka)

            3. Optymalizacja Kosztów:
            - Obecne koszty operacyjne
            - Strategie redukcji kosztów
            - Przewidywany czas zwrotu

            4. Usprawnienia Wydajności:
            - Obecne wąskie gardła
            - Zalecane usprawnienia
            - Oczekiwane zyski wydajnościowe
            </format_wyjściowy>

            <przykłady>
            UŻYTKOWNIK: Centrala wentylacyjna ma wydajność 2000 m³/h i pobór mocy 1.5 kW
            ASYSTENT: 
            1. Efektywność Energetyczna:
            - Obecne wskaźniki: 0.75 W/(m³/h)
            - Możliwość usprawnienia: Wymiana na model klasy IE4
            - Korzyści: Redukcja poboru mocy o 20%

            UŻYTKOWNIK: Brak danych o poborze mocy centrali
            ASYSTENT:
            1. Efektywność Energetyczna:
            - Obecne wskaźniki: BRAK DANYCH
            - Zalecenie: Konieczny audyt energetyczny
            - Korzyści: Nie można oszacować bez pomiarów
            </przykłady>

            Dokumentacja do analizy:
            {text}
            """
        return template.format(text=text)

    async def _arun(self, text: str) -> str:
        """Async version of the run method (not implemented)."""
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
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=Config.MODEL_TEMPERATURE,
            max_tokens=Config.MODEL_MAX_TOKENS,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.tools = [
            HVACTechnicalAnalysisTool(),
            HVACComplianceTool(),
            HVACOptimizationTool()
        ]
        
        template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Question: {input}
        {agent_scratchpad}
        
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=st.session_state.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

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

    def chunk_text(self, text: str) -> list[str]:
        """Dzieli tekst na mniejsze fragmenty z zachowaniem nakładania się."""
        if len(text) <= Config.CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + Config.CHUNK_SIZE
            
            # Jeśli to nie jest ostatni fragment, znajdź najbliższy koniec zdania
            if end < len(text):
                # Szukaj końca zdania w obszarze nakładania
                next_period = text.find('. ', end - Config.CHUNK_OVERLAP)
                if next_period != -1 and next_period < end + Config.CHUNK_OVERLAP:
                    end = next_period + 1
            
            chunks.append(text[start:end])
            start = end - Config.CHUNK_OVERLAP
        
        return chunks

    def merge_parameters(self, results: list[str]) -> str:
        """Łączy wyniki analizy z różnych fragmentów, zachowując podział na urządzenia."""
        devices = {}
        current_device = None
        
        for result in results:
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Sprawdź czy to nagłówek urządzenia
                if line.startswith('[') and line.endswith(']'):
                    current_device = line
                    if current_device not in devices:
                        devices[current_device] = set()
                # Jeśli to parametr, dodaj do aktualnego urządzenia
                elif line.startswith('-') and current_device:
                    devices[current_device].add(line)
        
        # Formatuj wynik końcowy
        result_parts = []
        for device, params in devices.items():
            result_parts.append(device)
            result_parts.extend(sorted(params))
            result_parts.append("")  # Pusta linia między urządzeniami
        
        return '\n'.join(result_parts)

    def analyze_document(self, doc_text: str, analysis_type: str):
        with st.spinner("Analizuję dokumentację..."):
            try:
                chunks = self.chunk_text(doc_text)
                all_results = []
                
                for i, chunk in enumerate(chunks):
                    with st.spinner(f"Analizuję część {i+1}/{len(chunks)}..."):
                        if analysis_type == "Pełna analiza":
                            result = self.agent_executor.invoke({
                                "input": f"""Extract and group technical parameters for all HVAC devices from the documentation.
                                Identify each separate device and list its parameters.

                                Documentation:
                                {chunk}

                                Required format:
                                [Device Name/ID]
                                - parameter: value (unit)
                                - parameter: value (unit)

                                [Next Device Name/ID]
                                - parameter: value (unit)
                                - parameter: value (unit)"""
                            })
                        else:
                            result = self.agent_executor.invoke({
                                "input": f"""Extract and group the most important technical parameters for all HVAC devices.
                                Identify each separate device and list its key parameters.

                                Documentation:
                                {chunk}

                                Required format:
                                [Device Name/ID]
                                - parameter: value (unit)
                                - parameter: value (unit)

                                [Next Device Name/ID]
                                - parameter: value (unit)
                                - parameter: value (unit)"""
                            })
                        all_results.append(result["output"])
                
                final_result = self.merge_parameters(all_results)
                
                st.success("Analiza zakończona!")
                st.write("### Parametry techniczne według urządzeń:")
                st.write(final_result)
                
                self.update_history(doc_text, final_result)
                
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
    