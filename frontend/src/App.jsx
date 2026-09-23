import { useCallback, useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

async function apiBlob(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.blob();
}

function HighlightedSnippet({ value }) {
  const parts = value.split(/(<mark>|<\/mark>)/);
  let highlighted = false;
  return parts.map((part, index) => {
    if (part === "<mark>") {
      highlighted = true;
      return null;
    }
    if (part === "</mark>") {
      highlighted = false;
      return null;
    }
    return highlighted ? <mark key={index}>{part}</mark> : <span key={index}>{part}</span>;
  });
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(
    new Date(value),
  );
}

function formatTimestamp(value) {
  const seconds = Math.max(0, Math.floor(value || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState("hybrid");
  const [workspaceMode, setWorkspaceMode] = useState("search");
  const [searchMeta, setSearchMeta] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [answerStatus, setAnswerStatus] = useState(null);
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);
  const [showImporter, setShowImporter] = useState(false);
  const [form, setForm] = useState({ title: "", content: "", source_type: "text", source_name: null });
  const [uploadFile, setUploadFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [voices, setVoices] = useState([]);
  const [voiceName, setVoiceName] = useState("");
  const [rate, setRate] = useState(1);
  const [speaking, setSpeaking] = useState(false);
  const [speechStatus, setSpeechStatus] = useState(null);
  const [audioPlayer, setAudioPlayer] = useState(null);
  const [embeddingStatus, setEmbeddingStatus] = useState(null);
  const [indexing, setIndexing] = useState(false);

  const loadDocuments = useCallback(async () => {
    const data = await api("/api/documents");
    setDocuments(data.items);
  }, []);

  const loadEmbeddingStatus = useCallback(async () => {
    const data = await api("/api/embeddings/status");
    setEmbeddingStatus(data);
  }, []);

  useEffect(() => {
    api("/health")
      .then((data) => setApiStatus(data.database === "connected" ? "online" : "degraded"))
      .catch(() => setApiStatus("offline"));
    loadDocuments().catch((err) => setError(err.message));
    loadEmbeddingStatus().catch((err) => setError(err.message));
    api("/api/answers/status").then(setAnswerStatus).catch(() => setAnswerStatus(null));
    api("/api/speech/status").then(setSpeechStatus).catch(() => setSpeechStatus(null));
  }, [loadDocuments, loadEmbeddingStatus]);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return undefined;
    const loadVoices = () => {
      const available = window.speechSynthesis.getVoices();
      setVoices(available);
      setVoiceName((current) => current || available.find((voice) => voice.default)?.name || available[0]?.name || "");
    };
    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      window.speechSynthesis.cancel();
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
    };
  }, []);

  useEffect(() => {
    if (speechStatus?.available && speechStatus.voices?.length) {
      setVoiceName((current) => speechStatus.voices.includes(current) ? current : speechStatus.voices[0]);
    }
  }, [speechStatus]);

  const visibleDocuments = searchMeta ? results : documents;
  const selectedVoice = useMemo(() => voices.find((voice) => voice.name === voiceName), [voices, voiceName]);
  const speechVoices = speechStatus?.available && speechStatus.voices?.length ? speechStatus.voices : voices.map((voice) => voice.name);

  async function runSearch(event) {
    event.preventDefault();
    const normalized = query.trim();
    if (!normalized) {
      setSearchMeta(null);
      setResults([]);
      setAnswer(null);
      return;
    }
    setBusy(true);
    setError("");
    try {
      if (workspaceMode === "ask") {
        const data = await api("/api/answers", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: normalized }),
        });
        setAnswer(data);
        setSearchMeta(null);
        setSelected(null);
        return;
      }
      const data = await api(`/api/search?q=${encodeURIComponent(normalized)}&mode=${searchMode}`);
      setAnswer(null);
      setResults(data.items);
      setSearchMeta({ total: data.total, elapsed: data.elapsed_ms, mode: data.mode, warning: data.warning });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function importFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadFile(file);
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      setForm({
        title: file.name.replace(/\.pdf$/i, ""),
        content: "",
        source_type: "pdf",
        source_name: file.name,
      });
      return;
    }
    if (file.type.startsWith("image/") || /\.(png|jpe?g|webp|tiff?|bmp)$/i.test(file.name)) {
      setForm({
        title: file.name.replace(/\.(png|jpe?g|webp|tiff?|bmp)$/i, ""),
        content: "",
        source_type: "image",
        source_name: file.name,
      });
      return;
    }
    if (file.type.startsWith("audio/") || /\.(mp3|wav|m4a|flac|ogg|aac|opus|aiff?)$/i.test(file.name)) {
      setForm({
        title: file.name.replace(/\.(mp3|wav|m4a|flac|ogg|aac|opus|aiff?)$/i, ""),
        content: "",
        source_type: "audio",
        source_name: file.name,
      });
      return;
    }
    if (file.type.startsWith("video/") || /\.(mp4|mov|mkv|webm|m4v)$/i.test(file.name)) {
      setForm({
        title: file.name.replace(/\.(mp4|mov|mkv|webm|m4v)$/i, ""),
        content: "",
        source_type: "video",
        source_name: file.name,
      });
      return;
    }
    const content = await file.text();
    setForm({
      title: file.name.replace(/\.(txt|md|markdown)$/i, ""),
      content,
      source_type: /\.(md|markdown)$/i.test(file.name) ? "markdown" : "text",
      source_name: file.name,
    });
  }

  async function saveDocument(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const binarySource = ["pdf", "image", "audio", "video"].includes(form.source_type);
      const uploadEndpoint = ["audio", "video"].includes(form.source_type) ? "media" : form.source_type;
      const created = binarySource && uploadFile
        ? await api(
            `/api/documents/${uploadEndpoint}?filename=${encodeURIComponent(uploadFile.name)}&title=${encodeURIComponent(form.title)}`,
            {
              method: "POST",
              headers: { "Content-Type": form.source_type === "pdf" ? "application/pdf" : (uploadFile.type || "application/octet-stream") },
              body: uploadFile,
            },
          )
        : await api("/api/documents", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(form),
          });
      setShowImporter(false);
      setForm({ title: "", content: "", source_type: "text", source_name: null });
      setUploadFile(null);
      setSelected(created);
      setSearchMeta(null);
      setResults([]);
      await loadDocuments();
      await loadEmbeddingStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteDocument(document) {
    if (!window.confirm(`Delete "${document.title}"?`)) return;
    try {
      await api(`/api/documents/${document.id}`, { method: "DELETE" });
      if (selected?.id === document.id) setSelected(null);
      setResults((current) => current.filter((item) => item.id !== document.id));
      await loadDocuments();
      await loadEmbeddingStatus();
    } catch (err) {
      setError(err.message);
    }
  }

  function speakInBrowser(document) {
    if (!("speechSynthesis" in window)) {
      setError("Text-to-speech is not supported by this browser.");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(`${document.title}. ${document.content}`);
    utterance.rate = rate;
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  }

  async function speak(document) {
    stopSpeaking();
    if (speechStatus?.available) {
      setBusy(true);
      setError("");
      try {
        const blob = await apiBlob("/api/speech", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: `${document.title}. ${document.content}`.slice(0, 20_000),
            voice: voiceName || null,
            rate: Math.round(rate * 180),
          }),
        });
        const url = URL.createObjectURL(blob);
        const player = new Audio(url);
        player.onended = () => {
          URL.revokeObjectURL(url);
          setSpeaking(false);
          setAudioPlayer(null);
        };
        player.onerror = () => {
          URL.revokeObjectURL(url);
          setSpeaking(false);
          setAudioPlayer(null);
          speakInBrowser(document);
        };
        setAudioPlayer(player);
        setSpeaking(true);
        await player.play();
      } catch (err) {
        speakInBrowser(document);
        setError(`Local speech unavailable, using browser voice. ${err.message}`);
      } finally {
        setBusy(false);
      }
      return;
    }
    speakInBrowser(document);
  }

  function stopSpeaking() {
    if (audioPlayer) {
      audioPlayer.pause();
      audioPlayer.currentTime = 0;
      setAudioPlayer(null);
    }
    window.speechSynthesis?.cancel();
    setSpeaking(false);
  }

  function openCitation(citation) {
    const document = documents.find((item) => item.id === citation.document_id);
    if (document) setSelected(document);
  }

  async function enableSemanticSearch() {
    setIndexing(true);
    setError("");
    try {
      const result = await api("/api/embeddings/reindex", { method: "POST" });
      setEmbeddingStatus(result);
      if (query.trim()) {
        const data = await api(`/api/search?q=${encodeURIComponent(query.trim())}&mode=${searchMode}`);
        setResults(data.items);
        setSearchMeta({ total: data.total, elapsed: data.elapsed_ms, mode: data.mode, warning: data.warning });
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => { setSelected(null); setSearchMeta(null); }}>
          <span className="brand-mark">N</span>
          <span>NexusAI</span>
        </button>
        <nav className="workspace-tabs" aria-label="Workspace mode">
          {["search", "ask"].map((mode) => (
            <button
              key={mode}
              className={workspaceMode === mode ? "active" : ""}
              onClick={() => { setWorkspaceMode(mode); setSelected(null); setAnswer(null); setSearchMeta(null); }}
            >
              {mode === "search" ? "Search" : "Ask AI"}
            </button>
          ))}
        </nav>
        <div className={`status status-${apiStatus}`}>
          <span className="status-dot" />
          {apiStatus}
        </div>
        <button className="primary-button" onClick={() => setShowImporter(true)}>Add document</button>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <span>Library</span>
            <span className="count">{documents.length}</span>
          </div>
          <nav className="document-nav" aria-label="Document library">
            {documents.map((document) => (
              <button
                className={selected?.id === document.id ? "nav-item active" : "nav-item"}
                key={document.id}
                onClick={() => setSelected(document)}
              >
                <span className="file-type">
                  {document.source_type === "markdown" ? "MD" : document.source_type === "pdf" ? "PDF" : document.source_type === "image" ? "IMG" : document.source_type === "audio" ? "AUD" : document.source_type === "video" ? "VID" : "TXT"}
                </span>
                <span className="nav-copy">
                  <strong>{document.title}</strong>
                  <small>{document.word_count.toLocaleString()} words</small>
                </span>
              </button>
            ))}
            {!documents.length && <p className="empty-sidebar">Your documents will appear here.</p>}
          </nav>
        </aside>

        <section className="content-area">
          <form className={`search-bar ${workspaceMode === "ask" ? "ask-bar" : ""}`} onSubmit={runSearch}>
            <div className="query-heading">
              <label htmlFor="search">{workspaceMode === "ask" ? "Ask your library" : "Search your knowledge"}</label>
              {workspaceMode === "ask" && <span className={answerStatus?.available ? "model-ready" : "model-offline"}>{answerStatus?.available ? `${answerStatus.model} ready` : "Local model offline"}</span>}
            </div>
            <div className="search-control">
              <input
                id="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={workspaceMode === "ask" ? "Ask a question answered from your documents" : "Try a phrase, topic, or exact term"}
              />
              {query && <button type="button" className="clear-button" onClick={() => { setQuery(""); setSearchMeta(null); setAnswer(null); }}>Clear</button>}
              <button className="search-button" disabled={busy}>{busy ? (workspaceMode === "ask" ? "Thinking" : "Searching") : (workspaceMode === "ask" ? "Ask" : "Search")}</button>
            </div>
            <div className={`search-options ${workspaceMode === "ask" ? "ask-options" : ""}`}>
              {workspaceMode === "search" && (
              <div className="mode-switch" aria-label="Search mode">
                {["hybrid", "keyword", "semantic"].map((mode) => (
                  <button
                    type="button"
                    key={mode}
                    className={searchMode === mode ? "active" : ""}
                    onClick={() => setSearchMode(mode)}
                  >
                    {mode}
                  </button>
                ))}
              </div>
              )}
              <div className="embedding-state">
                <span>
                  {embeddingStatus?.ready
                    ? `${embeddingStatus.indexed_chunks} passages indexed`
                    : embeddingStatus?.total_chunks
                      ? `${embeddingStatus.pending_chunks} passages need vectors`
                      : "Add documents to enable semantic search"}
                </span>
                {!!embeddingStatus?.total_chunks && !embeddingStatus.ready && (
                  <button type="button" onClick={enableSemanticSearch} disabled={indexing}>
                    {indexing ? "Indexing locally..." : "Enable semantic search"}
                  </button>
                )}
              </div>
            </div>
          </form>

          {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}

          {selected ? (
            <article className="reader">
              <div className="reader-header">
                <div>
                  <button className="back-button" onClick={() => setSelected(null)}>Back to results</button>
                  <h1>{selected.title}</h1>
                  <p>
                    {formatDate(selected.created_at)} / {selected.word_count.toLocaleString()} words / {selected.source_type}
                    {selected.source_type === "pdf" ? ` / ${selected.page_count} pages` : ""}
                    {selected.ocr_applied ? " / OCR" : ""}
                    {selected.duration_seconds ? ` / ${formatTimestamp(selected.duration_seconds)}` : ""}
                    {selected.language ? ` / ${selected.language.toUpperCase()}` : ""}
                  </p>
                </div>
                <button className="danger-button" onClick={() => deleteDocument(selected)}>Delete</button>
              </div>
              <div className="speech-panel">
                <button className="speech-button" onClick={() => speaking ? stopSpeaking() : speak(selected)}>
                  {speaking ? "Stop reading" : busy ? "Preparing" : "Read aloud"}
                </button>
                <label>
                  Voice
                  <select value={voiceName} onChange={(event) => setVoiceName(event.target.value)}>
                    {speechVoices.map((voice) => <option key={voice} value={voice}>{voice}</option>)}
                  </select>
                </label>
                <label>
                  Speed <output>{rate.toFixed(1)}x</output>
                  <input type="range" min="0.6" max="1.6" step="0.1" value={rate} onChange={(event) => setRate(Number(event.target.value))} />
                </label>
              </div>
              <div className="document-content">{selected.content}</div>
            </article>
          ) : answer ? (
            <section className="answer-view">
              <div className="answer-header">
                <p className="eyebrow">Grounded answer</p>
                <h1>{answer.question}</h1>
                <span>{answer.generated ? `Generated locally with ${answer.model}` : "Evidence mode"} / {answer.elapsed_ms} ms</span>
              </div>
              {answer.warning && <div className="search-warning">{answer.warning}</div>}
              <div className="answer-copy">{answer.answer}</div>
              <div className="evidence-heading"><h2>Sources</h2><span>{answer.citations.length} passages</span></div>
              <div className="evidence-list">
                {answer.citations.map((citation) => (
                  <button key={`${citation.document_id}-${citation.number}`} className="evidence-row" onClick={() => openCitation(citation)}>
                    <span className="citation-number">{citation.number}</span>
                    <span className="evidence-copy">
                      <strong>{citation.title}</strong>
                      <small>{citation.source_type}{citation.page_number ? ` / Page ${citation.page_number}` : ""}{citation.start_seconds != null ? ` / ${formatTimestamp(citation.start_seconds)} - ${formatTimestamp(citation.end_seconds)}` : ""}</small>
                      <span>{citation.passage}</span>
                    </span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <section className="results-view">
              <div className="results-heading">
                <div>
                  <p className="eyebrow">{searchMeta ? "Search results" : "Recent documents"}</p>
                  <h1>{searchMeta ? `Matches for "${query.trim()}"` : "Your local knowledge base"}</h1>
                </div>
                {searchMeta && <span className="search-stats">{searchMeta.total} results in {searchMeta.elapsed} ms</span>}
              </div>

              <div className="results-list">
                {searchMeta?.warning && <div className="search-warning">{searchMeta.warning}</div>}
                {visibleDocuments.map((document) => (
                  <article className="result-row" key={document.id}>
                    <button className="result-main" onClick={() => setSelected(document)}>
                      <div className="result-meta">
                        <span>{document.source_type}</span>
                        <span>{formatDate(document.created_at)}</span>
                        <span>{document.word_count.toLocaleString()} words</span>
                        {document.page_number && <span>Page {document.page_number}</span>}
                        {document.ocr_applied && <span>OCR</span>}
                        {document.start_seconds != null && <span>{formatTimestamp(document.start_seconds)} - {formatTimestamp(document.end_seconds)}</span>}
                      </div>
                      <h2>{document.title}</h2>
                      <p>{document.snippet ? <HighlightedSnippet value={document.snippet} /> : document.content.slice(0, 220)}</p>
                    </button>
                    <button className="read-quick" onClick={() => speak(document)}>Read</button>
                  </article>
                ))}
              </div>

              {!visibleDocuments.length && (
                <div className="empty-state">
                  <div className="empty-icon">N</div>
                  <h2>{searchMeta ? "No matching passages" : workspaceMode === "ask" ? "Ask across your library" : "Build your local library"}</h2>
                  <p>{searchMeta ? "Try fewer or more specific terms." : workspaceMode === "ask" ? "Your answer will stay grounded in indexed documents and show its sources." : "Add documents or media to make them searchable and readable aloud."}</p>
                  {!searchMeta && <button className="primary-button" onClick={() => setShowImporter(true)}>Add first document</button>}
                </div>
              )}
            </section>
          )}
        </section>
      </main>

      {showImporter && (
        <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowImporter(false); }}>
          <form className="import-modal" onSubmit={saveDocument}>
            <div className="modal-header">
              <div><p className="eyebrow">Local ingestion</p><h2>Add a document</h2></div>
              <button type="button" className="close-button" aria-label="Close" onClick={() => setShowImporter(false)}>Close</button>
            </div>
            <label className="file-drop">
              <strong>Choose a document, image, audio, or video file</strong>
              <span>The file stays on this machine.</span>
              <input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,.mp3,.wav,.m4a,.flac,.ogg,.aac,.opus,.aif,.aiff,.mp4,.mov,.mkv,.webm,.m4v,.txt,.md,.markdown,application/pdf,image/*,audio/*,video/*,text/plain,text/markdown" onChange={importFile} />
            </label>
            <div className="field-row">
              <label>Title<input required maxLength="240" value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label>
              <label>Format<select disabled={["pdf", "image", "audio", "video"].includes(form.source_type)} value={form.source_type} onChange={(event) => setForm({ ...form, source_type: event.target.value })}><option value="text">Plain text</option><option value="markdown">Markdown</option><option value="transcript">Transcript</option><option value="pdf">PDF</option><option value="image">Image OCR</option><option value="audio">Audio</option><option value="video">Video</option></select></label>
            </div>
            {["pdf", "image", "audio", "video"].includes(form.source_type) ? (
              <div className="pdf-ready"><strong>{uploadFile?.name}</strong><span>{["audio", "video"].includes(form.source_type) ? "Ready for local transcription and timestamp indexing." : "Ready for local extraction, OCR, and indexing."}</span></div>
            ) : (
              <label>Content<textarea required value={form.content} onChange={(event) => setForm({ ...form, content: event.target.value })} placeholder="Paste text here, or choose a file above." /></label>
            )}
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={() => setShowImporter(false)}>Cancel</button>
              <button className="primary-button" disabled={busy}>{busy ? (["audio", "video"].includes(form.source_type) ? "Transcribing" : "Indexing") : "Add and index"}</button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
