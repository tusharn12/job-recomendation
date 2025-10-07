"""Robust resume parser with PDF/DOCX support and structured extraction."""
import pdfplumber
from docx import Document
import re
from typing import Dict, List
from .utils import clean_text, simple_tokenize, logger

COMMON_SKILLS = [
    'python','java','c++','sql','aws','docker','kubernetes','nlp','tensorflow',
    'pytorch','scikit-learn','react','node','javascript','git','linux','bash',
    'excel','tableau','powerbi','spark','transformers'
]
EDU_KEYWORDS = ['bachelor', 'master', 'ph.d', 'b.sc', 'b.tech', 'mba', 'ms', 'bs']

class ParseError(Exception):
    pass

def _extract_text_from_pdf(path: str) -> str:
    try:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return clean_text('\n'.join(text))
    except Exception as e:
        logger.error(f"PDF parsing failed for {path}: {e}")
        raise ParseError(e)

def _extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return clean_text('\n'.join(paragraphs))
    except Exception as e:
        logger.error(f"DOCX parsing failed for {path}: {e}")
        raise ParseError(e)

def parse_resume(path: str) -> Dict:
    """Parse resume file and return structured dict: text, skills, education, experience."""
    if path.lower().endswith('.pdf'):
        full_text = _extract_text_from_pdf(path)
    elif path.lower().endswith('.docx'):
        full_text = _extract_text_from_docx(path)
    else:
        msg = "Unsupported resume format. Provide .pdf or .docx"
        logger.error(msg)
        raise ParseError(msg)

    skills = _extract_skills(full_text)
    education = _extract_education(full_text)
    experience = _extract_experience(full_text)

    return {
        'text': full_text,
        'skills': skills,
        'education': education,
        'experience': experience
    }

def _extract_skills(text: str) -> List[str]:
    tokens = simple_tokenize(text)
    found = set()
    lowered = text.lower()
    for skill in COMMON_SKILLS:
        if skill in lowered:
            found.add(skill)
    # token-level matches
    for t in tokens:
        if t in COMMON_SKILLS:
            found.add(t)
    return sorted(found)

def _extract_education(text: str) -> List[str]:
    lower = text.lower()
    found = []
    for kw in EDU_KEYWORDS:
        if kw in lower:
            m = re.search(r"([^.\n]{0,80}" + re.escape(kw) + r"[^.\n]{0,80})", lower)
            if m:
                found.append(m.group(0).strip())
    return found

def _extract_experience(text: str) -> List[str]:
    exps = []
    # years range like 2018-2021 or 2019 to 2022
    for m in re.finditer(r"(\b(19|20)\d{2})\s*(?:[-–to]{1,3})\s*(\b(19|20)\d{2})", text):
        exps.append(m.group(0))
    # lines mentioning experience/company/roles
    for line in text.split('\n'):
        ll = line.strip().lower()
        if 'experience' in ll or 'worked as' in ll or 'company' in ll or 'intern' in ll:
            exps.append(line.strip())
    return exps
