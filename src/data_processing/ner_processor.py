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


def save_ner_evaluation(entities,label_counts):
    NER_EVALUATION_PATH.parent.mkdir(parents=True,exist_ok=True)
    
    evaluation = {
        'total_entities':len(entities),
        'label_counts':label_counts
    }
    
    with open(NER_EVALUATION_PATH,'w',encoding='utf-8') as file:
        json.dump(evaluation,file,indent=4)
    print(f"NER evaluation saved to: {NER_EVALUATION_PATH}")

if __name__ == '__main__':
    main()