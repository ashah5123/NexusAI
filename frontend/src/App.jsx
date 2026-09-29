import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Bookmark,
  BookOpen,
  CheckCircle2,
  CheckSquare,
  FileAudio,
  FileImage,
  FileText,
  FileVideo,
  Download,
  Folder,
  Headphones,
  Highlighter,
  LayoutList,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Pencil,
  Plus,
  RefreshCw,
  ScanText,
  Search,
  Sparkles,
  Square,
  Star,
  StickyNote,
  Tags,
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

function AnnotatedContent({ content, highlights, onSelect, onHighlightClick, contentRef }) {
  const parts = [];
  let cursor = 0;
  for (const highlight of highlights) {
    if (highlight.start_offset < cursor || highlight.end_offset > content.length) continue;
    if (highlight.start_offset > cursor) parts.push(content.slice(cursor, highlight.start_offset));
    parts.push(
      <mark
        className={`reader-highlight highlight-${highlight.color}`}
        data-highlight-id={highlight.id}
        key={highlight.id}
        title={highlight.annotation || "Saved highlight"}
        onClick={() => onHighlightClick(highlight)}
      >
        {content.slice(highlight.start_offset, highlight.end_offset)}
      </mark>,
    );
    cursor = highlight.end_offset;
  }
  if (cursor < content.length) parts.push(content.slice(cursor));
  return <div className="document-content" ref={contentRef} onMouseUp={onSelect}>{parts}</div>;
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
  const [askCollection, setAskCollection] = useState("all");
  const [askSourceType, setAskSourceType] = useState("all");
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
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [mobileLibraryOpen, setMobileLibraryOpen] = useState(false);
  const [sourceLocation, setSourceLocation] = useState({ page: null, start: null, end: null });
  const [showMetadata, setShowMetadata] = useState(false);
  const [metadataForm, setMetadataForm] = useState({ title: "", collection: "", tags: "" });
  const [sortMode, setSortMode] = useState("recent");
  const [collectionFilter, setCollectionFilter] = useState("all");
  const [tagFilters, setTagFilters] = useState([]);
  const [dateFilter, setDateFilter] = useState("all");
  const [savedViews, setSavedViews] = useState([]);
  const [activeViewId, setActiveViewId] = useState("");
  const [bulkMode, setBulkMode] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [notes, setNotes] = useState([]);
  const [noteDraft, setNoteDraft] = useState("");
  const [editingNoteId, setEditingNoteId] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [highlightDraft, setHighlightDraft] = useState(null);
  const [highlightColor, setHighlightColor] = useState("yellow");
  const [highlightAnnotation, setHighlightAnnotation] = useState("");
  const [editingHighlightId, setEditingHighlightId] = useState(null);
  const [citationFocus, setCitationFocus] = useState(null);
  const completedJobs = useRef(new Set());
  const searchInput = useRef(null);
  const activeNotesDocument = useRef(null);
  const documentContentRef = useRef(null);

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

  const loadSavedViews = useCallback(async () => {
    const data = await api("/api/saved-views");
    setSavedViews(data.items);
  }, []);

  const loadNotes = useCallback(async (documentId) => {
    const data = await api(`/api/documents/${documentId}/notes`);
    if (activeNotesDocument.current === documentId) setNotes(data.items);
  }, []);

  const loadHighlights = useCallback(async (documentId) => {
    const data = await api(`/api/documents/${documentId}/highlights`);
    if (activeNotesDocument.current === documentId) setHighlights(data.items);
  }, []);

  useEffect(() => {
    api("/health")
      .then((data) => setApiStatus(data.database === "connected" ? "online" : "degraded"))
      .catch(() => setApiStatus("offline"));
    loadDocuments().catch((err) => setError(err.message));
    loadEmbeddingStatus().catch((err) => setError(err.message));
    loadIngestionJobs().catch((err) => setError(err.message));
    loadSavedViews().catch((err) => setError(err.message));
    api("/api/answers/status").then(setAnswerStatus).catch(() => setAnswerStatus(null));
    api("/api/speech/status").then(setSpeechStatus).catch(() => setSpeechStatus(null));
  }, [loadDocuments, loadEmbeddingStatus, loadIngestionJobs, loadSavedViews]);

  useEffect(() => {
    const saved = window.localStorage.getItem("nexusai-library-filters");
    if (!saved) return;
    try {
      const filters = JSON.parse(saved);
      setLibraryFilter(filters.source || "all");
      setFavoriteOnly(Boolean(filters.favorite));
      setCollectionFilter(filters.collection || "all");
      setTagFilters(Array.isArray(filters.tags) ? filters.tags : []);
      setDateFilter(filters.date || "all");
      setSortMode(filters.sort || "recent");
    } catch { /* Ignore stale local preferences. */ }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("nexusai-library-filters", JSON.stringify({
      source: libraryFilter, favorite: favoriteOnly, collection: collectionFilter,
      tags: tagFilters, date: dateFilter, sort: sortMode,
    }));
  }, [libraryFilter, favoriteOnly, collectionFilter, tagFilters, dateFilter, sortMode]);

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
        setShowMetadata(false);
        setMobileLibraryOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const selectedVoice = useMemo(() => voices.find((voice) => voice.name === voiceName), [voices, voiceName]);
  const speechVoices = speechStatus?.available && speechStatus.voices?.length ? speechStatus.voices : voices.map((voice) => voice.name);
  const activeJobs = ingestionJobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  const sourceUrl = selected ? `${API_BASE_URL}/api/documents/${selected.id}/source` : "";
  const collections = useMemo(() => [...new Set(documents.map((document) => document.collection).filter(Boolean))].sort(), [documents]);
  const allTags = useMemo(() => [...new Set(documents.flatMap((document) => document.tags))].sort(), [documents]);
  const filteredLibrary = useMemo(() => {
    const filtered = documents.filter((document) => {
      if (collectionFilter !== "all" && document.collection !== collectionFilter) return false;
      if (favoriteOnly && !document.favorite) return false;
      if (tagFilters.some((tag) => !document.tags.includes(tag))) return false;
      if (dateFilter !== "all") {
        const age = Date.now() - new Date(document.updated_at).getTime();
        const days = dateFilter === "7d" ? 7 : dateFilter === "30d" ? 30 : 365;
        if (age > days * 86_400_000) return false;
      }
      if (libraryFilter === "text") return ["text", "markdown", "transcript"].includes(document.source_type);
      if (libraryFilter === "scan") return ["pdf", "image"].includes(document.source_type);
      if (libraryFilter === "media") return ["audio", "video"].includes(document.source_type);
      return true;
    });
    return [...filtered].sort((left, right) => {
      if (sortMode === "title") return left.title.localeCompare(right.title);
      if (sortMode === "favorite") return Number(right.favorite) - Number(left.favorite) || left.title.localeCompare(right.title);
      return new Date(right.updated_at) - new Date(left.updated_at);
    });
  }, [documents, libraryFilter, favoriteOnly, collectionFilter, tagFilters, dateFilter, sortMode]);
  const visibleDocuments = searchMeta ? results : filteredLibrary;
  const readerHighlights = useMemo(() => {
    if (!citationFocus) return highlights;
    const focus = {
      id: "citation-focus",
      start_offset: citationFocus.start_offset,
      end_offset: citationFocus.end_offset,
      color: "citation",
      annotation: "Passage cited in the answer",
    };
    return [focus, ...highlights.filter((item) =>
      item.end_offset <= focus.start_offset || item.start_offset >= focus.end_offset
    )].sort((left, right) => left.start_offset - right.start_offset);
  }, [highlights, citationFocus]);

  useEffect(() => {
    if (!selected || !citationFocus) return;
    window.requestAnimationFrame(() => {
      document.querySelector('[data-highlight-id="citation-focus"]')?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
  }, [selected, citationFocus, highlights]);

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
          body: JSON.stringify({
            question: normalized,
            collection: askCollection === "all" ? null : askCollection,
            source_types: askSourceType === "all" ? [] : [askSourceType],
          }),
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
        openDocument(created);
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

  async function clearFinishedJobs() {
    try {
      await api("/api/ingestion-jobs", { method: "DELETE" });
      setIngestionJobs((current) => current.filter((job) => ["queued", "running"].includes(job.status)));
    } catch (err) {
      setError(err.message);
    }
  }

  function openDocument(document, location = {}) {
    setSelected(document);
    activeNotesDocument.current = document.id;
    setNotes([]);
    setNoteDraft("");
    setEditingNoteId(null);
    setHighlights([]);
    setHighlightDraft(null);
    setHighlightAnnotation("");
    setEditingHighlightId(null);
    setCitationFocus(
      location.focusStart != null && location.focusEnd != null
        ? { start_offset: location.focusStart, end_offset: location.focusEnd }
        : null,
    );
    loadNotes(document.id).catch((err) => setError(err.message));
    loadHighlights(document.id).catch((err) => setError(err.message));
    setSourceLocation({
      page: location.page ?? document.page_number ?? null,
      start: location.start ?? document.start_seconds ?? null,
      end: location.end ?? document.end_seconds ?? null,
    });
    setMobileLibraryOpen(false);
  }

  async function updateDocumentMetadata(document, changes) {
    const updated = await api(`/api/documents/${document.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    });
    setDocuments((current) => current.map((item) => item.id === updated.id ? updated : item));
    setResults((current) => current.map((item) => item.id === updated.id ? { ...item, ...updated } : item));
    setSelected((current) => current?.id === updated.id ? updated : current);
    return updated;
  }

  function sourceTypesForFilter(filter) {
    if (filter === "text") return ["text", "markdown", "transcript"];
    if (filter === "scan") return ["pdf", "image"];
    if (filter === "media") return ["audio", "video"];
    return [];
  }

  function filterForSourceTypes(sourceTypes) {
    const key = [...sourceTypes].sort().join(",");
    if (key === "markdown,text,transcript") return "text";
    if (key === "image,pdf") return "scan";
    if (key === "audio,video") return "media";
    return "all";
  }

  async function saveCurrentView() {
    const name = window.prompt("Name this smart view");
    if (!name?.trim()) return;
    try {
      const created = await api("/api/saved-views", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name, query: query.trim(), collection: collectionFilter === "all" ? null : collectionFilter,
          tags: tagFilters, source_types: sourceTypesForFilter(libraryFilter), favorite: favoriteOnly,
          date_range: dateFilter, sort: sortMode,
        }),
      });
      setSavedViews((current) => [...current, created].sort((a, b) => a.name.localeCompare(b.name)));
      setActiveViewId(created.id);
    } catch (err) { setError(err.message); }
  }

  function applySavedView(viewId) {
    setActiveViewId(viewId);
    const view = savedViews.find((item) => item.id === viewId);
    if (!view) return;
    setQuery(view.query);
    setCollectionFilter(view.collection || "all");
    setTagFilters(view.tags);
    setLibraryFilter(filterForSourceTypes(view.source_types));
    setFavoriteOnly(view.favorite);
    setDateFilter(view.date_range);
    setSortMode(view.sort);
    setSearchMeta(null);
    setAnswer(null);
  }

  async function removeSavedView() {
    if (!activeViewId || !window.confirm("Delete this saved view?")) return;
    try {
      await api(`/api/saved-views/${activeViewId}`, { method: "DELETE" });
      setSavedViews((current) => current.filter((view) => view.id !== activeViewId));
      setActiveViewId("");
    } catch (err) { setError(err.message); }
  }

  async function organizeSelected() {
    if (!selectedDocumentIds.length) return;
    const collection = window.prompt("Move selected documents to collection (leave blank to remove collection)", "");
    if (collection === null) return;
    const tags = window.prompt("Replace tags (comma separated; leave blank for none)", "");
    if (tags === null) return;
    try {
      await api("/api/documents", {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: selectedDocumentIds, collection, tags: tags.split(",").map((tag) => tag.trim()).filter(Boolean) }),
      });
      setSelectedDocumentIds([]);
      setBulkMode(false);
      await loadDocuments();
    } catch (err) { setError(err.message); }
  }

  async function renameCurrentCollection() {
    if (collectionFilter === "all") return;
    const name = window.prompt("Rename collection", collectionFilter);
    if (!name?.trim() || name.trim() === collectionFilter) return;
    try {
      await api(`/api/collections/${encodeURIComponent(collectionFilter)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
      });
      setCollectionFilter(name.trim());
      await loadDocuments();
    } catch (err) { setError(err.message); }
  }

  async function clearCurrentCollection() {
    if (collectionFilter === "all" || !window.confirm(`Remove collection “${collectionFilter}” from its documents?`)) return;
    try {
      await api(`/api/collections/${encodeURIComponent(collectionFilter)}`, { method: "DELETE" });
      setCollectionFilter("all");
      await loadDocuments();
    } catch (err) { setError(err.message); }
  }

  async function toggleFavorite(document) {
    try {
      await updateDocumentMetadata(document, { favorite: !document.favorite });
    } catch (err) {
      setError(err.message);
    }
  }

  async function saveNote(event) {
    event.preventDefault();
    if (!selected || !noteDraft.trim()) return;
    setBusy(true);
    setError("");
    try {
      if (editingNoteId) {
        const updated = await api(`/api/documents/${selected.id}/notes/${editingNoteId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: noteDraft }),
        });
        setNotes((current) => current.map((note) => note.id === updated.id ? updated : note));
      } else {
        const created = await api(`/api/documents/${selected.id}/notes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: noteDraft }),
        });
        setNotes((current) => [created, ...current]);
      }
      setNoteDraft("");
      setEditingNoteId(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function editNote(note) {
    setEditingNoteId(note.id);
    setNoteDraft(note.content);
  }

  async function deleteNote(note) {
    try {
      await api(`/api/documents/${note.document_id}/notes/${note.id}`, { method: "DELETE" });
      setNotes((current) => current.filter((item) => item.id !== note.id));
      if (editingNoteId === note.id) {
        setEditingNoteId(null);
        setNoteDraft("");
      }
    } catch (err) {
      setError(err.message);
    }
  }

  function captureHighlightSelection() {
    const root = documentContentRef.current;
    const selection = window.getSelection();
    if (!root || !selection || selection.rangeCount === 0 || selection.isCollapsed) return;
    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return;
    const beforeStart = range.cloneRange();
    beforeStart.selectNodeContents(root);
    beforeStart.setEnd(range.startContainer, range.startOffset);
    const beforeEnd = range.cloneRange();
    beforeEnd.selectNodeContents(root);
    beforeEnd.setEnd(range.endContainer, range.endOffset);
    const start = beforeStart.toString().length;
    const end = beforeEnd.toString().length;
    const selectedText = selected.content.slice(start, end);
    if (!selectedText.trim()) return;
    setEditingHighlightId(null);
    setHighlightDraft({ start_offset: start, end_offset: end, selected_text: selectedText });
    setHighlightAnnotation("");
  }

  async function saveHighlight(event) {
    event.preventDefault();
    if (!selected || !highlightDraft) return;
    setBusy(true);
    try {
      const saved = await api(`/api/documents/${selected.id}/highlights${editingHighlightId ? `/${editingHighlightId}` : ""}`, {
        method: editingHighlightId ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editingHighlightId
          ? { color: highlightColor, annotation: highlightAnnotation }
          : { ...highlightDraft, color: highlightColor, annotation: highlightAnnotation }),
      });
      setHighlights((current) => (editingHighlightId
        ? current.map((item) => item.id === saved.id ? saved : item)
        : [...current, saved].sort((a, b) => a.start_offset - b.start_offset)));
      setHighlightDraft(null);
      setHighlightAnnotation("");
      setEditingHighlightId(null);
      window.getSelection()?.removeAllRanges();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteHighlight(highlight) {
    try {
      await api(`/api/documents/${highlight.document_id}/highlights/${highlight.id}`, { method: "DELETE" });
      setHighlights((current) => current.filter((item) => item.id !== highlight.id));
      if (editingHighlightId === highlight.id) {
        setEditingHighlightId(null);
        setHighlightDraft(null);
        setHighlightAnnotation("");
      }
    } catch (err) {
      setError(err.message);
    }
  }

  function editHighlight(highlight) {
    setEditingHighlightId(highlight.id);
    setHighlightDraft({
      start_offset: highlight.start_offset,
      end_offset: highlight.end_offset,
      selected_text: highlight.selected_text,
    });
    setHighlightColor(highlight.color);
    setHighlightAnnotation(highlight.annotation || "");
  }

  function jumpToHighlight(highlight) {
    document.querySelector(`[data-highlight-id="${highlight.id}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function exportAnnotations() {
    if (!selected) return;
    const lines = [`# ${selected.title}`, "", "## Highlights", ""];
    highlights.forEach((highlight) => {
      lines.push(`> ${highlight.selected_text.replaceAll("\n", " ")}`);
      if (highlight.annotation) lines.push("", highlight.annotation);
      lines.push("", `<!-- offsets ${highlight.start_offset}-${highlight.end_offset}; color ${highlight.color} -->`, "");
    });
    lines.push("## Notes", "");
    notes.forEach((note) => lines.push(`- ${note.content.replaceAll("\n", " ")}`));
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selected.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "document"}-annotations.md`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function editMetadata(document) {
    setMetadataForm({
      title: document.title,
      collection: document.collection || "",
      tags: document.tags.join(", "),
    });
    setShowMetadata(true);
  }

  async function saveMetadata(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await updateDocumentMetadata(selected, {
        title: metadataForm.title,
        collection: metadataForm.collection || null,
        tags: metadataForm.tags.split(",").map((tag) => tag.trim()).filter(Boolean),
      });
      setShowMetadata(false);
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
      if (selected?.id === document.id) {
        setSelected(null);
        activeNotesDocument.current = null;
        setNotes([]);
        setSourceLocation({ page: null, start: null, end: null });
      }
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

  async function openCitation(citation) {
    try {
      const document = documents.find((item) => item.id === citation.document_id)
        || await api(`/api/documents/${citation.document_id}`);
      openDocument(document, {
        page: citation.page_number,
        start: citation.start_seconds,
        end: citation.end_seconds,
        focusStart: citation.start_offset,
        focusEnd: citation.end_offset,
      });
    } catch (err) {
      setError(err.message);
    }
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
        <button className="brand" onClick={() => { setSelected(null); activeNotesDocument.current = null; setNotes([]); setSourceLocation({ page: null, start: null, end: null }); setSearchMeta(null); }}>
          <span className="brand-mark">N</span>
          <span>NexusAI</span>
        </button>
        <nav className="workspace-tabs" aria-label="Workspace mode">
          <button className={workspaceMode === "search" ? "active" : ""} onClick={() => { setWorkspaceMode("search"); setSelected(null); activeNotesDocument.current = null; setNotes([]); setAnswer(null); setSearchMeta(null); }}><Search size={15} />Search</button>
          <button className={workspaceMode === "ask" ? "active" : ""} onClick={() => { setWorkspaceMode("ask"); setSelected(null); activeNotesDocument.current = null; setNotes([]); setAnswer(null); setSearchMeta(null); }}><MessageSquareText size={15} />Ask</button>
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
              ["favorite", "Favorites", Star],
              ["text", "Text", FileText],
              ["scan", "PDF and images", ScanText],
              ["media", "Audio and video", Headphones],
            ].map(([value, label, Icon]) => (
              <button key={value} className={(value === "favorite" ? favoriteOnly : libraryFilter === value) ? "active" : ""} title={label} aria-label={label} onClick={() => value === "favorite" ? setFavoriteOnly((current) => !current) : setLibraryFilter(value)}><Icon size={16} /></button>
            ))}
          </div>
          <div className="saved-view-controls">
            <label><Bookmark size={14} /><select aria-label="Saved smart view" value={activeViewId} onChange={(event) => applySavedView(event.target.value)}><option value="">Smart views</option>{savedViews.map((view) => <option key={view.id} value={view.id}>{view.name}</option>)}</select></label>
            <button title="Save current filters" aria-label="Save current filters" onClick={saveCurrentView}><Plus size={14} /></button>
            {activeViewId && <button title="Delete saved view" aria-label="Delete saved view" onClick={removeSavedView}><Trash2 size={13} /></button>}
          </div>
          <div className="library-controls">
            <label><Folder size={14} /><select aria-label="Filter by collection" value={collectionFilter} onChange={(event) => setCollectionFilter(event.target.value)}><option value="all">All collections</option>{collections.map((collection) => <option key={collection} value={collection}>{collection}</option>)}</select></label>
            <select aria-label="Sort library" value={sortMode} onChange={(event) => setSortMode(event.target.value)}><option value="recent">Recent</option><option value="title">Title</option><option value="favorite">Favorites</option></select>
            <label><Tags size={14} /><select aria-label="Add tag filter" value="" onChange={(event) => event.target.value && setTagFilters((current) => current.includes(event.target.value) ? current : [...current, event.target.value])}><option value="">Add tag</option>{allTags.filter((tag) => !tagFilters.includes(tag)).map((tag) => <option key={tag} value={tag}>{tag}</option>)}</select></label>
            <select aria-label="Filter by date" value={dateFilter} onChange={(event) => setDateFilter(event.target.value)}><option value="all">Any date</option><option value="7d">7 days</option><option value="30d">30 days</option><option value="year">1 year</option></select>
          </div>
          {!!tagFilters.length && <div className="tag-filter-list">{tagFilters.map((tag) => <button key={tag} onClick={() => setTagFilters((current) => current.filter((item) => item !== tag))}>{tag}<X size={11} /></button>)}</div>}
          <div className="organize-actions">
            {collectionFilter !== "all" && <><button onClick={renameCurrentCollection}><Pencil size={12} />Rename</button><button onClick={clearCurrentCollection}><Trash2 size={12} />Remove</button></>}
            <button className={bulkMode ? "active" : ""} onClick={() => { setBulkMode((current) => !current); setSelectedDocumentIds([]); }}><CheckSquare size={12} />{bulkMode ? "Done" : "Select"}</button>
          </div>
          {bulkMode && <div className="bulk-bar"><button onClick={() => setSelectedDocumentIds(filteredLibrary.map((document) => document.id))}>Select all</button><span>{selectedDocumentIds.length} selected</span><button className="bulk-apply" disabled={!selectedDocumentIds.length} onClick={organizeSelected}>Organize</button></div>}
          <nav className="document-nav" aria-label="Document library">
            {filteredLibrary.map((document) => (
              <div className="nav-select-row" key={document.id}>
                {bulkMode && <input type="checkbox" aria-label={`Select ${document.title}`} checked={selectedDocumentIds.includes(document.id)} onChange={() => setSelectedDocumentIds((current) => current.includes(document.id) ? current.filter((id) => id !== document.id) : [...current, document.id])} />}
                <button className={selected?.id === document.id ? "nav-item active" : "nav-item"} onClick={() => openDocument(document)}>
                  <span className={`source-icon source-${document.source_type}`}><SourceIcon type={document.source_type} /></span>
                  <span className="nav-copy"><strong>{document.favorite && <Star className="favorite-inline" size={11} fill="currentColor" />}{document.title}</strong><small>{document.collection || formatDate(document.created_at)} · {document.word_count.toLocaleString()} words</small></span>
                  <ArrowRight className="nav-arrow" size={15} />
                </button>
              </div>
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
              {workspaceMode === "ask" && <div className="ask-scope" aria-label="Answer scope"><label><Folder size={13} /><select value={askCollection} onChange={(event) => setAskCollection(event.target.value)}><option value="all">All collections</option>{collections.map((collection) => <option key={collection} value={collection}>{collection}</option>)}</select></label><label><FileText size={13} /><select value={askSourceType} onChange={(event) => setAskSourceType(event.target.value)}><option value="all">All source types</option><option value="text">Text</option><option value="markdown">Markdown</option><option value="pdf">PDF</option><option value="image">Images</option><option value="audio">Audio</option><option value="video">Video</option><option value="transcript">Transcripts</option></select></label></div>}
              <div className="embedding-state">
                {embeddingStatus?.ready ? <CheckCircle2 size={14} /> : <Sparkles size={14} />}
                <span>{embeddingStatus?.ready ? `${embeddingStatus.indexed_chunks} passages indexed` : embeddingStatus?.total_chunks ? `${embeddingStatus.pending_chunks} passages need vectors` : "Add documents to enable semantic search"}</span>
                {!!embeddingStatus?.total_chunks && !embeddingStatus.ready && <button type="button" onClick={enableSemanticSearch} disabled={indexing}>{indexing ? "Indexing locally" : "Enable semantic search"}</button>}
              </div>
            </div>
          </form>

          {!!ingestionJobs.length && showActivity && (
            <section className="job-queue" aria-label="Import activity" aria-live="polite">
              <div className="job-queue-heading"><div><Activity size={15} /><strong>Import activity</strong><span>{activeJobs} active</span></div><div className="job-heading-actions">{ingestionJobs.some((job) => ["completed", "failed", "cancelled"].includes(job.status)) && <button type="button" onClick={clearFinishedJobs}>Clear finished</button>}<button className="icon-button" title="Hide activity" aria-label="Hide activity" onClick={() => setShowActivity(false)}><X size={16} /></button></div></div>
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
                <div><button className="back-button" onClick={() => { setSelected(null); activeNotesDocument.current = null; setNotes([]); setSourceLocation({ page: null, start: null, end: null }); }}><ArrowLeft size={15} />Back</button><div className="reader-title"><span className={`source-icon source-${selected.source_type}`}><SourceIcon type={selected.source_type} size={19} /></span><h1>{selected.title}</h1></div><p>{formatDate(selected.created_at)} · {selected.word_count.toLocaleString()} words · {selected.source_type}{selected.source_type === "pdf" ? ` · ${selected.page_count} pages` : ""}{selected.ocr_applied ? " · OCR" : ""}{selected.duration_seconds ? ` · ${formatTimestamp(selected.duration_seconds)}` : ""}{selected.language ? ` · ${selected.language.toUpperCase()}` : ""}</p>{(selected.collection || selected.tags.length > 0) && <div className="metadata-line">{selected.collection && <span><Folder size={12} />{selected.collection}</span>}{selected.tags.map((tag) => <span key={tag}><Tags size={12} />{tag}</span>)}</div>}</div>
                <div className="reader-actions"><button className={`icon-button favorite-button ${selected.favorite ? "active" : ""}`} title={selected.favorite ? "Remove favorite" : "Add favorite"} aria-label={selected.favorite ? "Remove favorite" : "Add favorite"} onClick={() => toggleFavorite(selected)}><Star size={18} fill={selected.favorite ? "currentColor" : "none"} /></button><button className="icon-button edit-button" title="Edit details" aria-label="Edit document details" onClick={() => editMetadata(selected)}><Pencil size={17} /></button><button className="icon-button danger-button" title="Delete document" aria-label="Delete document" onClick={() => deleteDocument(selected)}><Trash2 size={18} /></button></div>
              </div>
              {selected.source_available && (
                <section className={`source-viewer viewer-${selected.source_type}`}>
                  <div className="source-viewer-heading"><div><SourceIcon type={selected.source_type} size={16} /><strong>Original source</strong></div>{sourceLocation.page && <span>Page {sourceLocation.page}</span>}{sourceLocation.start != null && <span>{formatTimestamp(sourceLocation.start)} – {formatTimestamp(sourceLocation.end)}</span>}</div>
                  {selected.source_type === "pdf" && <iframe title={`${selected.title} source`} src={`${sourceUrl}#page=${sourceLocation.page || 1}&view=FitH`} />}
                  {selected.source_type === "image" && <img src={sourceUrl} alt={selected.title} />}
                  {selected.source_type === "audio" && <audio key={`${selected.id}-${sourceLocation.start}`} controls preload="metadata" src={sourceUrl} onLoadedMetadata={(event) => { if (sourceLocation.start != null) event.currentTarget.currentTime = sourceLocation.start; }} />}
                  {selected.source_type === "video" && <video key={`${selected.id}-${sourceLocation.start}`} controls preload="metadata" src={sourceUrl} onLoadedMetadata={(event) => { if (sourceLocation.start != null) event.currentTarget.currentTime = sourceLocation.start; }} />}
                </section>
              )}
              <div className="speech-panel">
                <button className={`speech-button ${speaking ? "active" : ""}`} onClick={() => speaking ? stopSpeaking() : speak(selected)}>{speaking ? <Square size={16} /> : busy ? <LoaderCircle className="spin" size={17} /> : <Volume2 size={18} />}<span>{speaking ? "Stop reading" : busy ? "Preparing" : "Read aloud"}</span></button>
                <label>Voice<select value={voiceName} onChange={(event) => setVoiceName(event.target.value)}>{speechVoices.map((voice) => <option key={voice} value={voice}>{voice}</option>)}</select></label>
                <label>Speed <output>{rate.toFixed(1)}x</output><input type="range" min="0.6" max="1.6" step="0.1" value={rate} onChange={(event) => setRate(Number(event.target.value))} /></label>
              </div>
              <section className="highlights-panel" aria-label="Document highlights">
                <div className="notes-heading"><div><Highlighter size={16} /><h2>Highlights</h2><span>{highlights.length}</span></div><div className="highlight-heading-actions"><small>Select text below to highlight it</small><button type="button" title="Export notes and highlights" onClick={exportAnnotations}><Download size={14} />Export</button></div></div>
                {highlightDraft && (
                  <form className="highlight-form" onSubmit={saveHighlight}>
                    <blockquote>{highlightDraft.selected_text}</blockquote>
                    <div className="highlight-fields">
                      <label>Color<select value={highlightColor} onChange={(event) => setHighlightColor(event.target.value)}><option value="yellow">Yellow</option><option value="green">Green</option><option value="blue">Blue</option><option value="pink">Pink</option></select></label>
                      <label>Annotation<input value={highlightAnnotation} onChange={(event) => setHighlightAnnotation(event.target.value)} maxLength="10000" placeholder="Optional note about this passage" /></label>
                    </div>
                    <div><button type="button" className="secondary-button" onClick={() => { setHighlightDraft(null); setEditingHighlightId(null); setHighlightAnnotation(""); }}>Cancel</button><button className="primary-button" disabled={busy}><Highlighter size={16} />{editingHighlightId ? "Update highlight" : "Save highlight"}</button></div>
                  </form>
                )}
                {!!highlights.length && <div className="highlight-list">{highlights.map((highlight) => (
                  <article key={highlight.id} className={`highlight-card highlight-${highlight.color}`}>
                    <button type="button" className="highlight-jump" onClick={() => jumpToHighlight(highlight)}><q>{highlight.selected_text}</q>{highlight.annotation && <span>{highlight.annotation}</span>}</button>
                    <div className="highlight-actions"><button type="button" onClick={() => editHighlight(highlight)}>Edit</button><button type="button" className="highlight-delete" onClick={() => deleteHighlight(highlight)}>Delete</button></div>
                  </article>
                ))}</div>}
              </section>
              <section className="notes-panel" aria-label="Document notes">
                <div className="notes-heading"><div><StickyNote size={16} /><h2>Notes</h2><span>{notes.length}</span></div></div>
                <form className="note-form" onSubmit={saveNote}>
                  <textarea value={noteDraft} onChange={(event) => setNoteDraft(event.target.value)} placeholder="Capture a takeaway, follow-up, or source note." maxLength="10000" />
                  <div>
                    {editingNoteId && <button type="button" className="secondary-button" onClick={() => { setEditingNoteId(null); setNoteDraft(""); }}>Cancel edit</button>}
                    <button className="primary-button" disabled={busy || !noteDraft.trim()}>{busy ? <LoaderCircle className="spin" size={17} /> : <CheckCircle2 size={17} />}<span>{editingNoteId ? "Update note" : "Add note"}</span></button>
                  </div>
                </form>
                <div className="notes-list">
                  {notes.map((note) => (
                    <article className="note-row" key={note.id}>
                      <p>{note.content}</p>
                      <div><span>{formatDate(note.updated_at)}</span><button type="button" onClick={() => editNote(note)}>Edit</button><button type="button" onClick={() => deleteNote(note)}>Delete</button></div>
                    </article>
                  ))}
                  {!notes.length && <p className="empty-notes">No notes for this source yet.</p>}
                </div>
              </section>
              {citationFocus && <div className="citation-focus-banner"><CheckCircle2 size={15} />Showing the exact passage cited in your answer.<button type="button" onClick={() => setCitationFocus(null)}>Clear focus</button></div>}
              <AnnotatedContent content={selected.content} highlights={readerHighlights} contentRef={documentContentRef} onSelect={captureHighlightSelection} onHighlightClick={(highlight) => highlight.id !== "citation-focus" && jumpToHighlight(highlight)} />
            </article>
          ) : answer ? (
            <section className="answer-view view-enter">
              <div className="answer-header"><p className="eyebrow">Grounded answer</p><h1>{answer.question}</h1><span>{answer.grounded ? "Citations verified" : answer.generated ? `Generated locally with ${answer.model}` : "Evidence mode"} · {answer.scope_description} · {answer.elapsed_ms} ms</span></div>
              {answer.warning && <div className="search-warning">{answer.warning}</div>}
              <div className="answer-copy">{answer.answer}</div>
              <div className="evidence-heading"><h2>Sources</h2><span>{answer.citations.length} passages</span></div>
              <div className="evidence-list">{answer.citations.map((citation) => <button key={`${citation.document_id}-${citation.number}`} className="evidence-row" onClick={() => openCitation(citation)}><span className="citation-number">{citation.number}</span><span className="evidence-copy"><strong>{citation.title}</strong><small>{citation.source_type}{citation.page_number ? ` · Page ${citation.page_number}` : ""}{citation.start_seconds != null ? ` · ${formatTimestamp(citation.start_seconds)} – ${formatTimestamp(citation.end_seconds)}` : ""}{citation.start_offset != null && !citation.page_number && citation.start_seconds == null ? " · Exact passage" : ""} · score {citation.score.toFixed(3)}</small><span>{citation.passage}</span></span><ArrowRight size={17} /></button>)}</div>
            </section>
          ) : (
            <section className="results-view view-enter">
              <div className="results-heading"><div><p className="eyebrow">{searchMeta ? "Search results" : "Recent documents"}</p><h1>{searchMeta ? `Matches for “${query.trim()}”` : "Your local knowledge base"}</h1></div>{searchMeta && <span className="search-stats">{searchMeta.total} results · {searchMeta.elapsed} ms</span>}</div>
              <div className="results-list">
                {searchMeta?.warning && <div className="search-warning">{searchMeta.warning}</div>}
                {visibleDocuments.map((document) => <article className="result-row" key={document.id}><span className={`source-icon source-${document.source_type}`}><SourceIcon type={document.source_type} size={19} /></span><button className="result-main" onClick={() => openDocument(document)}><div className="result-meta"><span>{document.source_type}</span>{document.favorite && <span><Star size={9} fill="currentColor" />Favorite</span>}{document.collection && <span>{document.collection}</span>}<span>{formatDate(document.created_at)}</span><span>{document.word_count.toLocaleString()} words</span>{document.page_number && <span>Page {document.page_number}</span>}{document.ocr_applied && <span>OCR</span>}{document.start_seconds != null && <span>{formatTimestamp(document.start_seconds)} – {formatTimestamp(document.end_seconds)}</span>}</div><h2>{document.title}</h2><p>{document.snippet ? <HighlightedSnippet value={document.snippet} /> : document.content.slice(0, 220)}</p></button><button className="icon-button read-quick" title="Read aloud" aria-label={`Read ${document.title} aloud`} onClick={() => speak(document)}><Volume2 size={17} /></button><button className="icon-button open-result" title="Open document" aria-label={`Open ${document.title}`} onClick={() => openDocument(document)}><ArrowRight size={18} /></button></article>)}
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
      {showMetadata && selected && (
        <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowMetadata(false); }}>
          <form className="metadata-modal view-enter" onSubmit={saveMetadata}>
            <div className="modal-header"><div><p className="eyebrow">Document details</p><h2>Organize source</h2></div><button type="button" className="icon-button close-button" aria-label="Close" title="Close" onClick={() => setShowMetadata(false)}><X size={19} /></button></div>
            <label>Title<input required maxLength="240" value={metadataForm.title} onChange={(event) => setMetadataForm({ ...metadataForm, title: event.target.value })} /></label>
            <label>Collection<div className="input-with-icon"><Folder size={16} /><input maxLength="120" list="collection-options" value={metadataForm.collection} onChange={(event) => setMetadataForm({ ...metadataForm, collection: event.target.value })} /></div></label>
            <datalist id="collection-options">{collections.map((collection) => <option key={collection} value={collection} />)}</datalist>
            <label>Tags<div className="input-with-icon"><Tags size={16} /><input value={metadataForm.tags} onChange={(event) => setMetadataForm({ ...metadataForm, tags: event.target.value })} placeholder="research, planning, reference" /></div></label>
            <div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setShowMetadata(false)}>Cancel</button><button className="primary-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <CheckCircle2 size={17} />}Save details</button></div>
          </form>
        </div>
      )}
    </div>
  );
}
