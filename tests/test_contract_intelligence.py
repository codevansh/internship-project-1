from src.risk.contract_intelligence import (
    load_contract_intelligence_models,
    analyze_clause,
)


def main():
    print("CONTRACT INTELLIGENCE PRODUCTION TEST")

    tokenizer, model, id_to_label, nlp = (
        load_contract_intelligence_models()
    )

    clause = (
        "The supplier's liability shall not exceed $1,000,000. "
        "Either party may terminate this agreement with 60 days "
        "written notice, and the fees may increase by 10% annually."
    )

    result = analyze_clause(
        clause,
        tokenizer,
        model,
        id_to_label,
        nlp,
    )

    print("\nCLAUSE TYPE:")
    print(result["clause_type"])

    print("\nCONFIDENCE:")
    print(result["classification_confidence"])

    print("\nENTITIES:")
    for entity in result["entities"]:
        print(entity)

    print("\nCONTRACTUAL INDICATORS:")
    print(result["contractual_indicators"])

    print("\nTOP PREDICTIONS:")
    for item in result["top_predictions"]:
        print(
            f"{item['clause_type']} "
            f"({item['confidence']:.4f})"
        )


if __name__ == "__main__":
    main()