import PyPDF2
import io
from typing import List, Dict, Any
import fitz  # PyMuPDF
import re
from dataclasses import dataclass
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class HVACExtractedData(BaseModel):
    """Model for structured HVAC data extraction"""
    device_name: str = Field(description="Name or model of the HVAC device")
    airflow: Dict[str, Any] = Field(description="Air flow parameters with values and units")
    pressure: Dict[str, Any] = Field(description="Pressure parameters with values and units")
    power: Dict[str, Any] = Field(description="Power consumption parameters with values and units")
    efficiency: Dict[str, Any] = Field(description="Efficiency parameters with values and units")
    noise_level: Dict[str, Any] = Field(description="Noise level parameters with values and units")
    dimensions: Dict[str, Any] = Field(description="Physical dimensions with values and units")
    additional_params: Dict[str, Any] = Field(description="Any additional relevant parameters found")

class PDFProcessor:
    """Class for processing HVAC technical documentation PDFs using LLM"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.1
        )
        self.setup_extraction_chain()

    def setup_extraction_chain(self):
        """Setup the LLM chain for parameter extraction"""
        self.parser = PydanticOutputParser(pydantic_object=HVACExtractedData)
        
        template = """You are a specialized HVAC technical documentation analyzer. Your task is to extract technical parameters from the provided text.

        CONTEXT:
        The text comes from a PDF of HVAC technical documentation, which may be in Polish or English.
        The text might be unstructured and contain various technical specifications.

        TASK:
        Analyze the text and extract all relevant HVAC parameters into a structured format.

        TEXT TO ANALYZE:
        {text}

        REQUIREMENTS:
        1. Extract all numerical values with their units
        2. Group related parameters together
        3. Maintain original units as found in the text
        4. If a parameter has multiple values (e.g. min/max), include all
        5. For unclear or ambiguous values, add a note in additional_params

        {format_instructions}

        IMPORTANT:
        - Be precise with numbers and units
        - Include all found parameters
        - If a parameter is not found, set it to null
        - Convert any Polish technical terms to English but keep original values

        Extract the parameters and format them according to the specified schema:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract HVAC parameters from text using LLM"""
        try:
            # Prepare the prompt
            _input = self.prompt.format(text=text)
            
            # Get LLM response
            output = self.llm.invoke(_input)
            
            # Parse the response into structured data
            parsed_data = self.parser.parse(output.content)
            
            return parsed_data.dict()
            
        except Exception as e:
            raise Exception(f"Error extracting parameters: {str(e)}")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF file and return structured data"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = self.extract_text_from_pdf(pdf_path)
        parameters = self.extract_parameters(text)
        
        return parameters

    def batch_process_pdfs(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Process multiple PDF files from a directory"""
        results = []
        pdf_files = Path(pdf_directory).glob('*.pdf')
        
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(str(pdf_file))
                results.append({
                    'file_name': pdf_file.name,
                    'data': result
                })
            except Exception as e:
                results.append({
                    'file_name': pdf_file.name,
                    'error': str(e)
                })
                
        return results