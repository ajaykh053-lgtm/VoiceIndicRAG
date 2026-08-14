"""FastAPI Main Application for Multilingual Voice RAG System."""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root directory to sys.path so direct execution works
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.services.vector_store import vector_store
from app.chunking import chunk_document
from app.models.schemas import ChunkingStrategyEnum

# Setup Logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


def load_initial_samples_if_empty():
    """Seeds vector store with initial MSMARCO-XI multilingual sample data if empty."""
    if vector_store.total_count == 0:
        logger.info("Vector store is empty. Seeding initial multilingual sample passages...")
        samples_path = settings.SAMPLES_DATA_PATH
        if not samples_path.exists():
            # Create default multilingual sample dataset
            default_samples = [
                {
                    "doc_id": "doc_101",
                    "title": "भारत की राजधानी और भूगोल",
                    "passage_text": "नई दिल्ली भारत की आधिकारिक राजधानी और केंद्र शासित प्रदेश है। यह भारत सरकार की तीनों शाखाओं — कार्यपालिका, विधायिका और न्यायपालिका का केंद्र है। भारत एशिया महाद्वीप के दक्षिण में स्थित एक विशाल प्रायद्वीपीय देश है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_102",
                    "title": "ताजमहल का इतिहास",
                    "passage_text": "ताजमहल भारत के उत्तर प्रदेश राज्य के आगरा शहर में यमुना नदी के तट पर स्थित एक विश्व प्रसिद्ध संगमरमर का मकबरा है। इसे मुगल सम्राट शाहजहाँ ने अपनी प्रिय पत्नी मुमताज महल की याद में बनवाया था। यह यूनेस्को विश्व धरोहर स्थल है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_103",
                    "title": "सौर ऊर्जा और नवीकरणीय स्रोत",
                    "passage_text": "सौर ऊर्जा सूर्य से प्राप्त होने वाली स्वच्छ और अक्षय ऊर्जा का एक प्रमुख स्रोत है। सौर पैनल फोटोवोल्टिक प्रभाव के माध्यम से सूर्य के प्रकाश को सीधे बिजली में परिवर्तित करते हैं। यह कार्बन उत्सर्जन और वायु प्रदूषण को कम करने में सहायक है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_104",
                    "title": "कंप्यूटर मेमोरी और रैम",
                    "passage_text": "कंप्यूटर में मेमोरी डेटा और निर्देशों को संग्रहीत करने का माध्यम है। रैम (RAM - Random Access Memory) एक अस्थिर प्राथमिक मेमोरी है जो सक्रिय प्रोग्रामों के डेटा को तेजी से निष्पादित करने के लिए उपयोग की जाती है। रोम (ROM) स्थायी मेमोरी होती है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_105",
                    "title": "जल संरक्षण और वर्षा जल संचयन",
                    "passage_text": "जल संरक्षण भविष्य की पीढ़ियों के लिए स्वच्छ जल संसाधनों को संरक्षित करने की एक आवश्यक प्रक्रिया है। वर्षा जल संचयन (Rainwater Harvesting) तकनीक से भूजल स्तर को बढ़ाया जा सकता है और पानी की कमी को रोका जा सकता है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_106",
                    "title": "भारतीय संविधान और गणतंत्र दिवस",
                    "passage_text": "भारतीय संविधान 26 जनवरी 1950 को पूरे देश में लागू हुआ था। डॉ. भीमराव अंबेडकर को भारतीय संविधान का मुख्य निर्माता माना जाता है। भारत 26 जनवरी को गणतंत्र दिवस के रूप में मनाता है।",
                    "language": "hi"
                },
                {
                    "doc_id": "doc_107",
                    "title": "Artificial Intelligence & RAG",
                    "passage_text": "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances Large Language Models by retrieving relevant documents from external vector knowledge bases before generating responses, thereby reducing hallucinations and providing grounded context.",
                    "language": "en"
                }
            ]
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(default_samples, f, ensure_ascii=False, indent=2)

        # Read samples and chunk across multiple strategies
        try:
            with open(samples_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            all_chunks = []
            for doc in documents:
                for strat in [ChunkingStrategyEnum.METADATA_AWARE, ChunkingStrategyEnum.SENTENCE_SEMANTIC, ChunkingStrategyEnum.FIXED_SIZE, ChunkingStrategyEnum.HIERARCHICAL]:
                    chunks = chunk_document(
                        text=doc["passage_text"],
                        strategy=strat,
                        doc_id=doc["doc_id"],
                        language=doc.get("language", "hi"),
                        extra_metadata={"title": doc.get("title", "")}
                    )
                    all_chunks.extend(chunks)

            vector_store.add_chunks(all_chunks)
            vector_store.save()
            logger.info(f"Seeded {len(all_chunks)} chunks into FAISS vector index.")
        except Exception as e:
            logger.error(f"Failed to seed samples: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_initial_samples_if_empty()
    yield
    # Shutdown


app = FastAPI(
    title="Multilingual Voice RAG API (HH Goa 2026)",
    description="Production-grade Voice-Enabled Multilingual Indic Retrieval-Augmented Generation API with FAISS, Groq Llama-3.1, Sarvam STT, 4 Chunking Strategies & Guardrails Harness.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(api_router, prefix="/api/v1", tags=["RAG Pipeline"])

# Mount static frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", tags=["UI"])
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
