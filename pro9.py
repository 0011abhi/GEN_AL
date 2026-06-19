import re
import wikipediaapi
from pydantic import BaseModel, Field
from typing import List, Optional

class InstitutionDetails(BaseModel):
    name: str
    founder: Optional[str] = None
    founded: Optional[str] = None
    branches: List[str] = Field(default_factory=list)
    number_of_employees: Optional[int] = None
    summary: Optional[str] = None

name = input("Enter the Institution name: ")

wiki = wikipediaapi.Wikipedia(user_agent="InstitutionScraper/1.0", language='en')
page = wiki.page(name)
text = page.text

founder_match = re.search(r"(?:founded|established|started)\s+by\s+([^.\n,]+)", text, re.IGNORECASE)
founder = founder_match.group(1).strip() if founder_match else "Unknown"

year_match = re.search(r"(?:founded|established|started|incorporated)(?:\s+in)?\s+(\d{4})", text, re.IGNORECASE)
founded = year_match.group(1) if year_match else "Unknown"

branches_match = re.search(r"Branches\s*[:\-]?\s*(.*)", text, re.IGNORECASE)
branches = [b.strip() for b in branches_match.group(1).split(',')] if branches_match else []

employees_match = re.search(r"Number of employees\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
employee_count = int(employees_match.group(1).replace(',', '')) if employees_match else None

data = InstitutionDetails(
    name=page.title,
    founder=founder,
    founded=founded,
    branches=branches,
    number_of_employees=employee_count,
    summary=page.summary[:500] + "..."
)

print(data.model_dump_json(indent=2))
