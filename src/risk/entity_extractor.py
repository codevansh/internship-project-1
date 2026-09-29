import spacy

MODEL_NAME = "en_core_web_sm"


def load_ner_model():
    """ Load the spaCy English NER model. """
    return spacy.load(MODEL_NAME)


def extract_entities(nlp, text):
    """ Extract named entities from a contractual clause. Returns a structured list containing the entity text, entity label, and spaCy description. """
    doc = nlp(text)

    entities = []
    for entity in doc.ents:
        entities.append(
            {
                "text": entity.text,
                "label": entity.label_,
                "description": spacy.explain(entity.label_) or "",
            }
        )
    return entities
