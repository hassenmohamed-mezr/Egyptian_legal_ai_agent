import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from services.rag_service import RAGService


if __name__ == "__main__":
    service = RAGService()
    query = "ما هي شروط إنهاء عقد العمل حسب القانون المصري؟"

    try:
        result = service.get_answer(query)
        print("QUERY:", result["query"])
        print("\nANSWER:\n", result["answer"])
        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)
    except Exception as exc:
        print("ERROR:", exc)