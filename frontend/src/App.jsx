import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  FileAudio,
  FileImage,
  FileText,
  FileVideo,
  Headphones,
  LayoutList,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Plus,
  RefreshCw,
  ScanText,
  Search,
  Sparkles,
  Square,
  Trash2,
  UploadCloud,
  Volume2,
  X,
} from "lucide-react";

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

function SourceIcon({ type, size = 17 }) {
  if (type === "image") return <FileImage size={size} />;
  if (["audio", "transcript"].includes(type)) return <FileAudio size={size} />;
  if (type === "video") return <FileVideo size={size} />;
  return <FileText size={size} />;
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
  const [ingestionJobs, setIngestionJobs] = useState([]);
  const [showActivity, setShowActivity] = useState(true);
  const [libraryFilter, setLibraryFilter] = useState("all");
  const [mobileLibraryOpen, setMobileLibraryOpen] = useState(false);
  const completedJobs = useRef(new Set());
  const searchInput = useRef(null);

  const loadDocuments = useCallback(async () => {
    const data = await api("/api/documents");
    setDocuments(data.items);
  }, []);

  const loadEmbeddingStatus = useCallback(async () => {
    const data = await api("/api/embeddings/status");
    setEmbeddingStatus(data);
  }, []);

  const loadIngestionJobs = useCallback(async () => {
    const data = await api("/api/ingestion-jobs?limit=8");
    setIngestionJobs(data.items);
  }, []);

  useEffect(() => {
    api("/health")
      .then((data) => setApiStatus(data.database === "connected" ? "online" : "degraded"))
      .catch(() => setApiStatus("offline"));
    loadDocuments().catch((err) => setError(err.message));
    loadEmbeddingStatus().catch((err) => setError(err.message));
    loadIngestionJobs().catch((err) => setError(err.message));
    api("/api/answers/status").then(setAnswerStatus).catch(() => setAnswerStatus(null));
    api("/api/speech/status").then(setSpeechStatus).catch(() => setSpeechStatus(null));
  }, [loadDocuments, loadEmbeddingStatus, loadIngestionJobs]);

  useEffect(() => {
    if (!ingestionJobs.some((job) => ["queued", "running"].includes(job.status))) return undefined;
    const timer = window.setInterval(() => {
      loadIngestionJobs().catch((err) => setError(err.message));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [ingestionJobs, loadIngestionJobs]);

  useEffect(() => {
    const newlyCompleted = ingestionJobs.filter(
      (job) => job.status === "completed" && !completedJobs.current.has(job.id),
    );
    ingestionJobs.filter((job) => job.status === "completed").forEach((job) => completedJobs.current.add(job.id));
    if (newlyCompleted.length) {
      loadDocuments().catch((err) => setError(err.message));
      loadEmbeddingStatus().catch((err) => setError(err.message));
    }
  }, [ingestionJobs, loadDocuments, loadEmbeddingStatus]);

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

  useEffect(() => {
    const handleKeyDown = (event) => {
      const target = event.target;
      const isEditing = ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName);
      if (event.key === "/" && !isEditing) {
        event.preventDefault();
        searchInput.current?.focus();
      }
      if (event.key === "Escape") {
        setShowImporter(false);
        setMobileLibraryOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const visibleDocuments = searchMeta ? results : documents;
  const selectedVoice = useMemo(() => voices.find((voice) => voice.name === voiceName), [voices, voiceName]);
  const speechVoices = speechStatus?.available && speechStatus.voices?.length ? speechStatus.voices : voices.map((voice) => voice.name);
  const activeJobs = ingestionJobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  const filteredLibrary = useMemo(() => documents.filter((document) => {
    if (libraryFilter === "text") return ["text", "markdown", "transcript"].includes(document.source_type);
    if (libraryFilter === "scan") return ["pdf", "image"].includes(document.source_type);
    if (libraryFilter === "media") return ["audio", "video"].includes(document.source_type);
    return true;
  }), [documents, libraryFilter]);

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
      setSearchMeta(null);
      setResults([]);
      if (binarySource) {
        setIngestionJobs((current) => [created, ...current.filter((job) => job.id !== created.id)]);
      } else {
        setSelected(created);
        await loadDocuments();
        await loadEmbeddingStatus();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function updateIngestionJob(job, action) {
    try {
      const updated = await api(`/api/ingestion-jobs/${job.id}/${action}`, { method: "POST" });
      setIngestionJobs((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (err) {
      setError(err.message);
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
        <button className="icon-button mobile-menu" title="Open library" aria-label="Open library" onClick={() => setMobileLibraryOpen(true)}><Menu size={19} /></button>
        <button className="brand" onClick={() => { setSelected(null); setSearchMeta(null); }}>
          <span className="brand-mark">N</span>
          <span>NexusAI</span>
        </button>
        <nav className="workspace-tabs" aria-label="Workspace mode">
          <button className={workspaceMode === "search" ? "active" : ""} onClick={() => { setWorkspaceMode("search"); setSelected(null); setAnswer(null); setSearchMeta(null); }}><Search size={15} />Search</button>
          <button className={workspaceMode === "ask" ? "active" : ""} onClick={() => { setWorkspaceMode("ask"); setSelected(null); setAnswer(null); setSearchMeta(null); }}><MessageSquareText size={15} />Ask</button>
        </nav>
        {!!ingestionJobs.length && (
          <button className={`icon-button activity-button ${showActivity ? "active" : ""}`} title="Import activity" aria-label="Toggle import activity" onClick={() => setShowActivity((current) => !current)}>
            <Activity size={18} />
            {!!activeJobs && <span>{activeJobs}</span>}
          </button>
        )}
        <div className={`status status-${apiStatus}`} title={`Service ${apiStatus}`}><span className="status-dot" />{apiStatus}</div>
        <button className="primary-button add-button" onClick={() => setShowImporter(true)}><Plus size={17} /><span>Add document</span></button>
      </header>

      <main className="workspace">
        {mobileLibraryOpen && <button className="sidebar-scrim" aria-label="Close library" onClick={() => setMobileLibraryOpen(false)} />}
        <aside className={`sidebar ${mobileLibraryOpen ? "mobile-open" : ""}`}>
          <div className="sidebar-heading">
            <div><BookOpen size={17} /><span>Library</span><span className="count">{documents.length}</span></div>
            <button className="icon-button sidebar-close" title="Close library" aria-label="Close library" onClick={() => setMobileLibraryOpen(false)}><X size={18} /></button>
          </div>
          <div className="library-filters" aria-label="Filter library">
            {[
              ["all", "All sources", LayoutList],
              ["text", "Text", FileText],
              ["scan", "PDF and images", ScanText],
              ["media", "Audio and video", Headphones],
            ].map(([value, label, Icon]) => (
              <button key={value} className={libraryFilter === value ? "active" : ""} title={label} aria-label={label} onClick={() => setLibraryFilter(value)}><Icon size={16} /></button>
            ))}
          </div>
          <nav className="document-nav" aria-label="Document library">
            {filteredLibrary.map((document) => (
              <button className={selected?.id === document.id ? "nav-item active" : "nav-item"} key={document.id} onClick={() => { setSelected(document); setMobileLibraryOpen(false); }}>
                <span className={`source-icon source-${document.source_type}`}><SourceIcon type={document.source_type} /></span>
                <span className="nav-copy"><strong>{document.title}</strong><small>{formatDate(document.created_at)} · {document.word_count.toLocaleString()} words</small></span>
                <ArrowRight className="nav-arrow" size={15} />
              </button>
            ))}
            {!filteredLibrary.length && <p className="empty-sidebar">No documents in this view.</p>}
          </nav>
        </aside>

        <section className="content-area">
          <form className={`search-bar ${workspaceMode === "ask" ? "ask-bar" : ""}`} onSubmit={runSearch}>
            <div className="query-heading">
              <label htmlFor="search">{workspaceMode === "ask" ? "Ask your library" : "Search your knowledge"}</label>
              {workspaceMode === "ask" && <span className={answerStatus?.available ? "model-ready" : "model-offline"}>{answerStatus?.available ? `${answerStatus.model} ready` : "Local model offline"}</span>}
            </div>
            <div className="search-control">
              <Search className="search-leading" size={19} />
              <input ref={searchInput} id="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={workspaceMode === "ask" ? "Ask a question answered from your documents" : "Try a phrase, topic, or exact term"} />
              {query && <button type="button" className="clear-button" title="Clear query" aria-label="Clear query" onClick={() => { setQuery(""); setSearchMeta(null); setAnswer(null); searchInput.current?.focus(); }}><X size={17} /></button>}
              <button className="search-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : workspaceMode === "ask" ? <Sparkles size={17} /> : <Search size={17} />}<span>{busy ? "Working" : workspaceMode === "ask" ? "Ask" : "Search"}</span></button>
            </div>
            <div className={`search-options ${workspaceMode === "ask" ? "ask-options" : ""}`}>
              {workspaceMode === "search" && <div className="mode-switch" aria-label="Search mode">{["hybrid", "keyword", "semantic"].map((mode) => <button type="button" key={mode} className={searchMode === mode ? "active" : ""} onClick={() => setSearchMode(mode)}>{mode}</button>)}</div>}
              <div className="embedding-state">
                {embeddingStatus?.ready ? <CheckCircle2 size={14} /> : <Sparkles size={14} />}
                <span>{embeddingStatus?.ready ? `${embeddingStatus.indexed_chunks} passages indexed` : embeddingStatus?.total_chunks ? `${embeddingStatus.pending_chunks} passages need vectors` : "Add documents to enable semantic search"}</span>
                {!!embeddingStatus?.total_chunks && !embeddingStatus.ready && <button type="button" onClick={enableSemanticSearch} disabled={indexing}>{indexing ? "Indexing locally" : "Enable semantic search"}</button>}
              </div>
            </div>
          </form>

          {!!ingestionJobs.length && showActivity && (
            <section className="job-queue" aria-label="Import activity" aria-live="polite">
              <div className="job-queue-heading"><div><Activity size={15} /><strong>Import activity</strong><span>{activeJobs} active</span></div><button className="icon-button" title="Hide activity" aria-label="Hide activity" onClick={() => setShowActivity(false)}><X size={16} /></button></div>
              <div className="job-list">
                {ingestionJobs.slice(0, 4).map((job) => (
                  <div className={`job-row job-${job.status}`} key={job.id}>
                    <span className={`source-icon source-${job.source_type}`}><SourceIcon type={job.source_type} /></span>
                    <div className="job-copy"><div><strong>{job.title}</strong><span>{job.stage}</span></div><div className="job-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={job.progress}><span style={{ width: `${job.progress}%` }} /></div>{job.error && <small>{job.error}</small>}</div>
                    <span className="job-status">{job.status}</span>
                    {["queued", "running"].includes(job.status) && <button className="icon-button job-action" type="button" title="Cancel import" aria-label="Cancel import" onClick={() => updateIngestionJob(job, "cancel")}><Square size={14} /></button>}
                    {["failed", "cancelled"].includes(job.status) && <button className="icon-button job-action" type="button" title="Retry import" aria-label="Retry import" onClick={() => updateIngestionJob(job, "retry")}><RefreshCw size={15} /></button>}
                    {job.status === "completed" && <CheckCircle2 className="job-complete" size={17} />}
                  </div>
                ))}
              </div>
            </section>
          )}

          {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" title="Dismiss" aria-label="Dismiss" onClick={() => setError("")}><X size={17} /></button></div>}

          {selected ? (
            <article className="reader view-enter">
              <div className="reader-header">
                <div><button className="back-button" onClick={() => setSelected(null)}><ArrowLeft size={15} />Back</button><div className="reader-title"><span className={`source-icon source-${selected.source_type}`}><SourceIcon type={selected.source_type} size={19} /></span><h1>{selected.title}</h1></div><p>{formatDate(selected.created_at)} · {selected.word_count.toLocaleString()} words · {selected.source_type}{selected.source_type === "pdf" ? ` · ${selected.page_count} pages` : ""}{selected.ocr_applied ? " · OCR" : ""}{selected.duration_seconds ? ` · ${formatTimestamp(selected.duration_seconds)}` : ""}{selected.language ? ` · ${selected.language.toUpperCase()}` : ""}</p></div>
                <button className="icon-button danger-button" title="Delete document" aria-label="Delete document" onClick={() => deleteDocument(selected)}><Trash2 size={18} /></button>
              </div>
              <div className="speech-panel">
                <button className={`speech-button ${speaking ? "active" : ""}`} onClick={() => speaking ? stopSpeaking() : speak(selected)}>{speaking ? <Square size={16} /> : busy ? <LoaderCircle className="spin" size={17} /> : <Volume2 size={18} />}<span>{speaking ? "Stop reading" : busy ? "Preparing" : "Read aloud"}</span></button>
                <label>Voice<select value={voiceName} onChange={(event) => setVoiceName(event.target.value)}>{speechVoices.map((voice) => <option key={voice} value={voice}>{voice}</option>)}</select></label>
                <label>Speed <output>{rate.toFixed(1)}x</output><input type="range" min="0.6" max="1.6" step="0.1" value={rate} onChange={(event) => setRate(Number(event.target.value))} /></label>
              </div>
              <div className="document-content">{selected.content}</div>
            </article>
          ) : answer ? (
            <section className="answer-view view-enter">
              <div className="answer-header"><p className="eyebrow">Grounded answer</p><h1>{answer.question}</h1><span>{answer.generated ? `Generated locally with ${answer.model}` : "Evidence mode"} · {answer.elapsed_ms} ms</span></div>
              {answer.warning && <div className="search-warning">{answer.warning}</div>}
              <div className="answer-copy">{answer.answer}</div>
              <div className="evidence-heading"><h2>Sources</h2><span>{answer.citations.length} passages</span></div>
              <div className="evidence-list">{answer.citations.map((citation) => <button key={`${citation.document_id}-${citation.number}`} className="evidence-row" onClick={() => openCitation(citation)}><span className="citation-number">{citation.number}</span><span className="evidence-copy"><strong>{citation.title}</strong><small>{citation.source_type}{citation.page_number ? ` · Page ${citation.page_number}` : ""}{citation.start_seconds != null ? ` · ${formatTimestamp(citation.start_seconds)} – ${formatTimestamp(citation.end_seconds)}` : ""}</small><span>{citation.passage}</span></span><ArrowRight size={17} /></button>)}</div>
            </section>
          ) : (
            <section className="results-view view-enter">
              <div className="results-heading"><div><p className="eyebrow">{searchMeta ? "Search results" : "Recent documents"}</p><h1>{searchMeta ? `Matches for “${query.trim()}”` : "Your local knowledge base"}</h1></div>{searchMeta && <span className="search-stats">{searchMeta.total} results · {searchMeta.elapsed} ms</span>}</div>
              <div className="results-list">
                {searchMeta?.warning && <div className="search-warning">{searchMeta.warning}</div>}
                {visibleDocuments.map((document) => <article className="result-row" key={document.id}><span className={`source-icon source-${document.source_type}`}><SourceIcon type={document.source_type} size={19} /></span><button className="result-main" onClick={() => setSelected(document)}><div className="result-meta"><span>{document.source_type}</span><span>{formatDate(document.created_at)}</span><span>{document.word_count.toLocaleString()} words</span>{document.page_number && <span>Page {document.page_number}</span>}{document.ocr_applied && <span>OCR</span>}{document.start_seconds != null && <span>{formatTimestamp(document.start_seconds)} – {formatTimestamp(document.end_seconds)}</span>}</div><h2>{document.title}</h2><p>{document.snippet ? <HighlightedSnippet value={document.snippet} /> : document.content.slice(0, 220)}</p></button><button className="icon-button read-quick" title="Read aloud" aria-label={`Read ${document.title} aloud`} onClick={() => speak(document)}><Volume2 size={17} /></button><button className="icon-button open-result" title="Open document" aria-label={`Open ${document.title}`} onClick={() => setSelected(document)}><ArrowRight size={18} /></button></article>)}
              </div>
              {!visibleDocuments.length && <div className="empty-state"><div className="empty-icon">{searchMeta ? <Search size={27} /> : <BookOpen size={27} />}</div><h2>{searchMeta ? "No matching passages" : workspaceMode === "ask" ? "Ask across your library" : "Build your local library"}</h2><p>{searchMeta ? "Try fewer or more specific terms." : workspaceMode === "ask" ? "Your answer will stay grounded in indexed documents and show its sources." : "Add documents or media to make them searchable and readable aloud."}</p>{!searchMeta && <button className="primary-button" onClick={() => setShowImporter(true)}><Plus size={17} />Add first document</button>}</div>}
            </section>
          )}
        </section>
      </main>

      {showImporter && (
        <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowImporter(false); }}>
          <form className="import-modal view-enter" onSubmit={saveDocument}>
            <div className="modal-header"><div><p className="eyebrow">Local ingestion</p><h2>Add a document</h2></div><button type="button" className="icon-button close-button" aria-label="Close" title="Close" onClick={() => setShowImporter(false)}><X size={19} /></button></div>
            <label className={`file-drop ${uploadFile ? "has-file" : ""}`}><UploadCloud size={28} /><strong>{uploadFile ? uploadFile.name : "Choose a document, image, audio, or video"}</strong><span>{uploadFile ? `${(uploadFile.size / 1024 / 1024).toFixed(1)} MB · stored locally` : "Browse files from this machine"}</span><input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp,.mp3,.wav,.m4a,.flac,.ogg,.aac,.opus,.aif,.aiff,.mp4,.mov,.mkv,.webm,.m4v,.txt,.md,.markdown,application/pdf,image/*,audio/*,video/*,text/plain,text/markdown" onChange={importFile} /></label>
            <div className="field-row"><label>Title<input required maxLength="240" value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label><label>Format<select disabled={["pdf", "image", "audio", "video"].includes(form.source_type)} value={form.source_type} onChange={(event) => setForm({ ...form, source_type: event.target.value })}><option value="text">Plain text</option><option value="markdown">Markdown</option><option value="transcript">Transcript</option><option value="pdf">PDF</option><option value="image">Image OCR</option><option value="audio">Audio</option><option value="video">Video</option></select></label></div>
            {["pdf", "image", "audio", "video"].includes(form.source_type) ? <div className="pdf-ready"><CheckCircle2 size={17} /><div><strong>{uploadFile?.name}</strong><span>{["audio", "video"].includes(form.source_type) ? "Ready for transcription and timestamp indexing." : "Ready for extraction, OCR, and indexing."}</span></div></div> : <label>Content<textarea required value={form.content} onChange={(event) => setForm({ ...form, content: event.target.value })} placeholder="Paste text here, or choose a file above." /></label>}
            <div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setShowImporter(false)}>Cancel</button><button className="primary-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <UploadCloud size={17} />}<span>{busy ? "Uploading" : ["pdf", "image", "audio", "video"].includes(form.source_type) ? "Add to queue" : "Add and index"}</span></button></div>
          </form>
        </div>
      )}
    </div>
  );
}
