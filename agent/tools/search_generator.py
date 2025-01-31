from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from agent.model import load_model
prompt_text = """
**Role**: You are a search engine optimization expert specializing in Russian educational institutions. Your queries consistently achieve top Google rankings.

**Task**: Generate 3-5 precise search queries in the original question's language (may be Russian) to best answer:

"{question}"

**Technical Requirements**:
1. Language Preservation:
   - Maintain original question's language (especially Russian)
   - Preserve Cyrillic characters and Russian punctuation

2. Keyword Strategy:
   - Enclose exact phrases in [square brackets]
   - Use official names ("ITMO University" not "ИТМО универ")
   - Include years when mentioned in question
   - Avoid interrogatives (how, what, кто, какой)

3. Search Operators:
   - Use site:itmo.ru for official information
   - Apply filetype:pdf/doc for documents
   - Specify date ranges (1995..2000)
   - Use quotes for exact matches

**Examples**:

Q: Сколько факультетов существует в Университете ИТМО?
A: 1. [факультеты Университета ИТМО] site:itmo.ru
   2. "количество факультетов" filetype:pdf ИТМО
   3. структура университета ИТМО официальный сайт

Q: Кто из сотрудников университета получил премию правительства РФ в области образования в 2016 году?
A: 1. [премия правительства РФ 2016] лауреаты образование site:itmo.ru
   2. награжденные сотрудники ИТМО 2016 filetype:doc
   3. "правительственная премия в области образования" 2016 список

Q: Какой научный центр был создан в ИТМО в 1995 году?
A: 1. [научные центры ИТМО] создание 1995..1997
   2. история научных подразделений site:itmo.ru
   3. "основан в 1995 году" научный центр filetype:pdf

**Output Rules**:
- Numeric list only (1. ...)
- 1 query per line
- Preserve original Cyrillic characters
- No translations to English
- Use Russian search operators (e.g., "файлtype:pdf" instead of "filetype:pdf" when needed)
"""

class SearchQueryGeneratorInput(BaseModel):
    original_query: str = Field(description="The original user question or request")

class SearchQueryGeneratorTool(BaseTool):
    name: str = "search_query_generator"
    description: str = """Генерирует несколько вариантов запросов исходя из предоставленного вопроса. 
    ОБЯЗАТЕЛЬНО использовать перед поисковыми запросами.
    Входом должен быть оригинальный вопрос."""
    args_schema: Type[BaseModel] = SearchQueryGeneratorInput

    def _run(self, original_query: str) -> List[str]:
        """Генерирует 3-5 поисковых запросов по следующим правилам:
        1. Сохраняет оригинальный язык вопроса
        2. Разбивает сложные вопросы на аспекты
        3. Использует ключевые слова из вопроса
        4. Добавляет модификаторы для точного поиска
        """
        
        # Initialize model
        llm = load_model()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("human", "{question}")
        ])
        
        # Create chain
        chain = prompt | llm
        
        # Generate queries
        response = chain.invoke({"question": original_query})
        
        # Parse response
        queries = response.content.split(", ")
        return queries[:5]  # Return max 5 queries
