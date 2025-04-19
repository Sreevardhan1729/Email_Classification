import re
import spacy
from typing import List, Dict, Tuple

nlp = spacy.load("en_core_web_sm")

PATTERNS = {
    'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    'phone_number': r'\b\d{10}\b',
    'dob': r'\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:\d{4})\b',
    'aadhar_num': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    'credit_debit_no': r'\b\d{4}(?:[-\s]?\d{4}){3}\b',
    'cvv_no': r'\b\d{3}\b',
    'expiry_no': r'\b(?:0[1-9]|1[0-2])/[0-9]{2}\b'
}

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('PERSON',):
            start, end = ent.start_char, ent.end_char
            entities.append({'position': [start, end], 'classification': 'full_name', 'entity': ent.text})
    for label, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            s, e = match.span()
            entities.append({'position': [s, e], 'classification': label, 'entity': match.group()})
    entities.sort(key=lambda x: x['position'][0], reverse=True)
    masked = text
    for ent in entities:
        s, e = ent['position']
        masked = masked[:s] + f"[{ent['classification']}]" + masked[e:]
    return masked, entities


def demask_pii(masked_text: str, entities: List[Dict]) -> str:
    for ent in sorted(entities, key=lambda x: x['position'][0]):
        token = f"[{ent['classification']}]"
        masked_text = masked_text.replace(token, ent['entity'], 1)
    return masked_text