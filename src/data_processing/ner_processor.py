from pathlib import Path
import spacy
import json

OCR_INPUT_PATH = Path("data/processed/ocr_sample.txt")
NER_EVALUATION_PATH= Path('data/processed/ner_evaluation.json')


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
    
    label_counts=label_entities(entities)
    for label,count in label_counts.items():
        print(f"{label}:{count}")
    
    display_entities(entities)
    print("\n Baseline NER processing completed successfully")
    
    save_ner_evaluation(entities,label_counts)
    
    
# Day-7 --> NER (Named Entities Recognition)
def label_entities(entities):
    label_counts={}
    for i in entities:
        label = i['label']
        
        if label in label_counts:
            label_counts[label]+=1
        else:
            label_counts[label]=1
    return label_counts


def create_error_examples():
    error_examples = [
        {
            "text": "HOFV",
            "predicted": "PERSON",
            "issue": "Organization/person confusion"
        },
        {
            "text": "Constellation NewEnergy",
            "predicted": "PERSON",
            "issue": "Organization/person confusion"
        },
        {
            "text": "Constellation",
            "predicted": "PRODUCT",
            "issue": "Company name classified as product"
        },
        {
            "text": "Agreement",
            "predicted": "PRODUCT",
            "issue": "Legal terminology misclassified"
        },
        {
            "text": "Exhibit C-1",
            "predicted": "PERSON",
            "issue": "Document heading misclassified"
        },
        {
            "text": "Johnson Controls",
            "predicted": "PERSON",
            "issue": "Organization/person confusion"
        },
        {
            "text": "Tom Benson Stadium",
            "predicted": "PERSON",
            "issue": "Facility/location misclassified"
        },
        {
            "text": "Section 4.2(d",
            "predicted": "DATE",
            "issue": "Contract section misclassified as date"
        }
    ]

    return error_examples


def create_error_patterns():
    error_patterns = [
        "Organization/person confusion",
        "Legal terminology misclassification",
        "Document heading misclassification",
        "Facility/location misclassification",
        "OCR-related text fragmentation"
    ]

    return error_patterns


def save_ner_evaluation(entities,label_counts):
    NER_EVALUATION_PATH.parent.mkdir(parents=True,exist_ok=True)
    
    error_examples = create_error_examples()
    error_patterns = create_error_patterns()

    evaluation = {
        'total_entities': len(entities),
        'label_counts': label_counts,
        'error_examples': error_examples,
        'error_patterns': error_patterns
    }
    
    with open(NER_EVALUATION_PATH,'w',encoding='utf-8') as file:
        json.dump(evaluation,file,indent=4)
    print(f"NER evaluation saved to: {NER_EVALUATION_PATH}")


if __name__ == '__main__':
    main()