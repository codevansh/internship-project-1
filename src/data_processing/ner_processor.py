from pathlib import Path
import spacy

OCR_INPUT_PATH = Path("data/processed/ocr_sample.txt")

def load_text(file_path):
    if not file_path.exists():
        raise FileNotFoundError("OCR text not found at")
    
    with open(file_path,"r",encoding='utf-8') as file:
        text= file.read()
        
    print("OCR text loaded")
    print("Characters loaded")
    
    return text


def load_ner_model():
    nlp=spacy.load("en_core_web_sm")
    return nlp


def extract_entities(nlp,text):
    doc=nlp(text)
    entities=[]
    
    for entity in doc.ents:
        entities.append({
            "text":entity.text,
            'label':entity.label_,
            "description":spacy.explain(entity.label_)
        })
        
    return entities
    

def display_entities(entities):
    print(f"\n Total Entites found: {len(entities)}")
    
    for entity in entities:
        print(
            f"{entity['text']} to "
            f"{entity['label']} "
            f"{entity['description']} "
        )


def main():
    text=load_text(OCR_INPUT_PATH)
    nlp=load_ner_model()
    entities=extract_entities(nlp,text)
    display_entities(entities)
    print("\n Baseline NER processing completed successfully")
    
if __name__ == '__main__':
    main()