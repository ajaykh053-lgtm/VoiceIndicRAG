/**
 * Voice Indic RAG - Client Application Logic
 * Integrates Web Audio recording, FastAPI backend endpoints, and Latency Analytics.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const micBtn = document.getElementById("mic-btn");
    const voiceStatusLabel = document.getElementById("voice-status-label");
    const recordingTimer = document.getElementById("recording-timer");
    const textQueryInput = document.getElementById("text-query-input");
    const sendQueryBtn = document.getElementById("send-query-btn");
    const strategySelect = document.getElementById("strategy-select");
    const languageSelect = document.getElementById("language-select");
    const answerBox = document.getElementById("answer-box");
    const modelTag = document.getElementById("model-tag");
    const guardIndicator = document.getElementById("guardrail-indicator");
    const guardText = document.getElementById("guard-text");
    const totalChunksCount = document.getElementById("total-chunks-count");
    const contextList = document.getElementById("context-list");
    const contextCount = document.getElementById("context-count");
    const contextsToggle = document.getElementById("contexts-toggle");

    // Latency Elements
    const latStt = document.getElementById("lat-stt");
    const latRetrieval = document.getElementById("lat-retrieval");
    const latLlm = document.getElementById("lat-llm");
    const latTotal = document.getElementById("lat-total");

    // Benchmark Modal Elements
    const runBenchmarkBtn = document.getElementById("run-benchmark-btn");
    const benchmarkModal = document.getElementById("benchmark-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const benchLoading = document.getElementById("bench-loading");
    const benchContent = document.getElementById("bench-content");
    const p50Val = document.getElementById("p50-val");
    const p70Val = document.getElementById("p70-val");
    const p90Val = document.getElementById("p90-val");
    const p100Val = document.getElementById("p100-val");
    const stageAvgContent = document.getElementById("stage-avg-content");

    // State
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let recordSeconds = 0;

    // 1. Initial Health & Stats Check
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            totalChunksCount.textContent = data.faiss_total_chunks || "0";
            if (data.status === "healthy") {
                document.getElementById("status-text").textContent = "Active (" + (data.groq_configured ? "Groq Live" : "Local Mode") + ")";
            }
        } catch (e) {
            console.warn("Health check error:", e);
            totalChunksCount.textContent = "Loaded";
        }
    }
    checkHealth();

    // 2. Audio Recording (Web Audio API)
    micBtn.addEventListener("click", async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                stream.getTracks().forEach(track => track.stop());
                await sendVoiceQuery(audioBlob);
            };

            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add("recording");
            voiceStatusLabel.textContent = "Listening... Speak now";
            
            recordSeconds = 0;
            recordingTimer.textContent = "00:00";
            timerInterval = setInterval(() => {
                recordSeconds++;
                const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
                const secs = String(recordSeconds % 60).padStart(2, "0");
                recordingTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            console.error("Microphone access denied or error:", err);
            voiceStatusLabel.textContent = "Microphone access error. Please type your query.";
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            micBtn.classList.remove("recording");
            clearInterval(timerInterval);
            voiceStatusLabel.textContent = "Processing speech with Sarvam STT & Groq...";
        }
    }

    // 3. Send Voice Query to Backend
    async function sendVoiceQuery(audioBlob) {
        setLoadingState(true);
        try {
            const formData = new FormData();
            formData.append("file", audioBlob, "query_voice.wav");
            formData.append("language", languageSelect.value === "hi" ? "hi-IN" : "en-IN");
            formData.append("strategy", strategySelect.value);
            formData.append("top_k", 4);

            const res = await fetch("/api/v1/voice", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`);
            }

            const data = await res.json();
            textQueryInput.value = data.transcription || "";
            voiceStatusLabel.textContent = `Heard: "${data.transcription}"`;
            renderRAGResponse(data.rag);

        } catch (err) {
            console.error("Voice RAG error:", err);
            answerBox.textContent = "Voice processing error. Please try again or use text search.";
            voiceStatusLabel.textContent = "Click Microphone to speak your query";
        } finally {
            setLoadingState(false);
        }
    }

    // 4. Send Text Query to Backend
    sendQueryBtn.addEventListener("click", () => {
        executeTextQuery(textQueryInput.value.trim());
    });

    textQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            executeTextQuery(textQueryInput.value.trim());
        }
    });

    // Sample Chips Click Handler
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            textQueryInput.value = q;
            executeTextQuery(q);
        });
    });

    async function executeTextQuery(query) {
        if (!query) return;
        setLoadingState(true);
        voiceStatusLabel.textContent = "Running RAG query...";

        try {
            const res = await fetch("/api/v1/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    language: languageSelect.value,
                    strategy: strategySelect.value,
                    top_k: 4
                })
            });

            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`);
            }

            const data = await res.json();
            renderRAGResponse(data);
            voiceStatusLabel.textContent = "Click Microphone to speak your query";

        } catch (err) {
            console.error("Text RAG error:", err);
            answerBox.textContent = "Error communicating with RAG backend. Please verify server is running.";
        } finally {
            setLoadingState(false);
        }
    }

    // 5. Render Response & Latency Metrics
    function renderRAGResponse(rag) {
        // Answer & Model
        answerBox.textContent = rag.answer || "No response generated.";
        modelTag.textContent = rag.model_used || "Llama-3.1-8B";

        // Latency
        const lat = rag.latency || {};
        latStt.textContent = (lat.stt_ms || 0.0).toFixed(1) + " ms";
        latRetrieval.textContent = (lat.retrieval_ms || 0.0).toFixed(1) + " ms";
        latLlm.textContent = (lat.llm_generation_ms || 0.0).toFixed(1) + " ms";
        latTotal.textContent = (lat.total_e2e_ms || 0.0).toFixed(1) + " ms";

        // Guardrails
        const guards = rag.guardrails || {};
        if (!guards.is_safe || !guards.is_on_topic || guards.is_hallucination) {
            guardIndicator.classList.add("flagged");
            guardText.textContent = `Guardrail Alert: ${guards.flag_reason || "Check Flagged"}`;
        } else {
            guardIndicator.classList.remove("flagged");
            const groundPercent = Math.round((rag.groundedness_score || 1.0) * 100);
            guardText.textContent = `🛡️ Guardrails: Safe & Grounded (${groundPercent}%)`;
        }

        // Retrieved Context Cards
        contextList.innerHTML = "";
        const contexts = rag.retrieved_contexts || [];
        contextCount.textContent = contexts.length;

        contexts.forEach((c, idx) => {
            const card = document.createElement("div");
            card.className = "context-card";
            card.innerHTML = `
                <div class="context-card-meta">
                    <span>#${idx + 1} • Chunk: ${c.chunk_id} [${c.metadata?.strategy || 'default'}]</span>
                    <span class="score-tag">Similarity: ${(c.score || 0.0).toFixed(3)}</span>
                </div>
                <div>${escapeHTML(c.text)}</div>
            `;
            contextList.appendChild(card);
        });
    }

    function setLoadingState(loading) {
        if (loading) {
            answerBox.textContent = "Retrieving passages from FAISS and generating grounded answer with Groq Llama 3.1...";
            sendQueryBtn.disabled = true;
        } else {
            sendQueryBtn.disabled = false;
        }
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }

    // Toggle Contexts accordion
    contextsToggle.addEventListener("click", () => {
        contextList.classList.toggle("hidden");
    });

    // 6. Latency Percentiles Benchmark (P50/P70/P100)
    runBenchmarkBtn.addEventListener("click", async () => {
        benchmarkModal.classList.remove("hidden");
        benchLoading.classList.remove("hidden");
        benchContent.classList.add("hidden");

        try {
            const res = await fetch("/api/v1/benchmark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    num_queries: 20,
                    strategy: strategySelect.value
                })
            });

            const data = await res.json();
            const p = data.latency_percentiles;

            p50Val.textContent = p.p50.toFixed(1) + " ms";
            p70Val.textContent = p.p70.toFixed(1) + " ms";
            p90Val.textContent = p.p90.toFixed(1) + " ms";
            p100Val.textContent = p.p100.toFixed(1) + " ms";

            const avgs = data.stage_averages || {};
            stageAvgContent.innerHTML = `
                <p><strong>Total Runs:</strong> ${data.num_queries_run} sample queries | <strong>Total Benchmark Time:</strong> ${(data.total_time_ms / 1000).toFixed(2)}s</p>
                <p><strong>Stage Averages:</strong></p>
                <ul>
                    <li>Guardrails Pre-Check: ${(avgs.guardrails_pre_avg_ms || 0).toFixed(1)} ms</li>
                    <li>FAISS Dense Vector Retrieval: ${(avgs.retrieval_avg_ms || 0).toFixed(1)} ms</li>
                    <li>Groq Llama 3.1 LLM Generation: ${(avgs.llm_generation_avg_ms || 0).toFixed(1)} ms</li>
                    <li>Hallucination Post-Check: ${(avgs.guardrails_post_avg_ms || 0).toFixed(1)} ms</li>
                </ul>
            `;

            benchLoading.classList.add("hidden");
            benchContent.classList.remove("hidden");

        } catch (e) {
            console.error("Benchmark error:", e);
            benchLoading.innerHTML = "<p>Error executing benchmark. Please try again.</p>";
        }
    });

    closeModalBtn.addEventListener("click", () => {
        benchmarkModal.classList.add("hidden");
    });
});
