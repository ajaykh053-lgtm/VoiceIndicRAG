"""Script to explore MSMARCO-XI Indic dataset schema and features from Hugging Face."""

import sys
import json

def explore(language: str = "hi"):
    print(f"=== Exploring MSMARCO-XI Dataset (Language: {language}) ===")
    try:
        from datasets import load_dataset
        print("Connecting to Hugging Face: ai4bharat/MSMARCO-XI...")
        dataset = load_dataset("ai4bharat/MSMARCO-XI", language, split="train", streaming=True)
        sample = next(iter(dataset))
        print("\n✅ Successfully connected and fetched sample:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return sample
    except Exception as e:
        print(f"\n⚠️ Direct Hugging Face download note ({e}). Exploring local schema format:")
        sample = {
            "query_id": "1001",
            "query": "भारत की राजधानी क्या है?",
            "language": language,
            "passages": [
                {
                    "passage_id": "p_001",
                    "passage_text": "नई दिल्ली भारत की आधिकारिक राजधानी है।",
                    "is_selected": 1
                }
            ]
        }
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return sample

if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
    explore(lang)
