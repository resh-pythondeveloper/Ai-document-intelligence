import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api.js";

const INDEXED_KEY = "indexedDocs";

function loadIndexed() {
  try {
    return new Set(JSON.parse(localStorage.getItem(INDEXED_KEY) || "[]"));
  } catch {
    return new Set();
  }
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [indexed, setIndexed] = useState(loadIndexed);
  const [busyDoc, setBusyDoc] = useState(null);
  const [scope, setScope] = useState(null); // null = all documents
  const [mode, setMode] = useState("standard");
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [notice, setNotice] = useState(null);
  const [drawer, setDrawer] = useState(false);
  const [deletingDoc, setDeletingDoc] = useState(null);
  const fileRef = useRef(null);
  const endRef = useRef(null);

  const flash = (text, tone = "error") => setNotice({ text, tone });

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.listDocuments());
    } catch (e) {
      flash(e.message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, thinking]);
  useEffect(() => {
    try { localStorage.setItem(INDEXED_KEY, JSON.stringify([...indexed])); } catch {}
  }, [indexed]);

  async function onUpload(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setBusyDoc("upload");
    try {
      const doc = await api.uploadDocument(file);
      await refresh();
      flash(`${doc.filename} uploaded. Index it so you can ask questions about it.`, "info");
    } catch (err) {
      flash(err.message);
    } finally {
      setBusyDoc(null);
    }
  }

  async function onDelete(doc) {
    setDeletingDoc(doc);
  }

  async function confirmDelete() {
    if (!deletingDoc) return;
    const doc = deletingDoc;
    setDeletingDoc(null);
    setBusyDoc(`delete-${doc.id}`);
    try {
      const result = await api.deleteDocument(doc.id);
      setIndexed((prev) => {
        const next = new Set(prev);
        next.delete(doc.id);
        return next;
      });
      if (scope === doc.id) setScope(null);
      await refresh();
      flash(`${doc.filename} deleted. ${result.vectors_deleted} vectors removed from Qdrant.`, "info");
    } catch (err) {
      flash(err.message);
    } finally {
      setBusyDoc(null);
    }
  }

  async function onIndex(doc) {
    setBusyDoc(doc.id);
    try {
      const r = await api.indexDocument(doc.id);
      setIndexed((prev) => new Set(prev).add(doc.id));
      flash(`${doc.filename} indexed: ${r.pages} pages, ${r.chunks} passages.`, "info");
    } catch (err) {
      flash(err.message);
    } finally {
      setBusyDoc(null);
    }
  }

  async function onSend(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || thinking) return;
    setInput("");
    setNotice(null);
    setMessages((m) => [...m, { role: "user", text: question }]);
    setThinking(true);
    try {
      if (mode === "standard") {
        const r = await api.askStandard({ question, documentId: scope, sessionId });
        setSessionId(r.session_id);
        setMessages((m) => [...m, { role: "assistant", text: r.answer, sources: r.sources }]);
      } else {
        const r = await api.askAgentic({ question, documentId: scope });
        setMessages((m) => [...m, {
          role: "assistant", text: r.answer, sources: r.sources,
          route: r.route, rewritten: r.rewritten_question,
        }]);
      }
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", error: err.message }]);
    } finally {
      setThinking(false);
    }
  }

  function newChat() {
    setMessages([]);
    setSessionId(null);
    setNotice(null);
  }

  const scopeName = scope ? docs.find((d) => d.id === scope)?.title ?? "selected document" : "all documents";

  return (
    <div className="app">
      <aside className={`side ${drawer ? "open" : ""}`} aria-label="Documents">
        <div className="side-head">
          <h1>Document Intelligence</h1>
          <button className="icon-btn close" onClick={() => setDrawer(false)} aria-label="Close documents">✕</button>
        </div>

        <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={onUpload} />
        <button className="primary block" onClick={() => fileRef.current.click()} disabled={busyDoc === "upload"}>
          {busyDoc === "upload" ? "Uploading…" : "Upload PDF"}
        </button>

        <div className="scope" role="radiogroup" aria-label="Search scope">
          <label className={scope === null ? "on" : ""}>
            <input type="radio" name="scope" checked={scope === null} onChange={() => setScope(null)} />
            All documents
          </label>
        </div>

        <ul className="docs">
          {docs.length === 0 && <li className="empty">No documents yet. Upload a PDF to get started.</li>}
          {docs.map((d) => (
            <li key={d.id} className={scope === d.id ? "on" : ""}>
              <label>
                <input type="radio" name="scope" checked={scope === d.id} onChange={() => setScope(d.id)} />
                <span className="doc-name" title={d.filename}>{d.title || d.filename}</span>
              </label>
              <div className="doc-actions">
                {indexed.has(d.id) ? (
                  <span className="tag ok">Indexed</span>
                ) : (
                  <button className="ghost" onClick={() => onIndex(d)} disabled={busyDoc === d.id || busyDoc === `delete-${d.id}`}>
                    {busyDoc === d.id ? "Indexing…" : "Index"}
                  </button>
                )}
                <button
                  className="danger-icon"
                  onClick={() => onDelete(d)}
                  disabled={busyDoc === `delete-${d.id}`}
                  aria-label={`Delete ${d.filename}`}
                  title="Delete document and its vectors"
                >
                  {busyDoc === `delete-${d.id}` ? "…" : "🗑"}
                </button>
              </div>
            </li>
          ))}
        </ul>
      </aside>
      {drawer && <div className="scrim" onClick={() => setDrawer(false)} />}

      {deletingDoc && (
        <div className="modal-backdrop" role="presentation" onClick={() => setDeletingDoc(null)}>
          <div className="delete-modal" role="dialog" aria-modal="true" aria-labelledby="delete-title" onClick={(e) => e.stopPropagation()}>
            <h2 id="delete-title">Delete document?</h2>
            <p>
              <strong>{deletingDoc.filename}</strong> will be removed from PostgreSQL, the uploaded PDF, and its Qdrant vectors.
            </p>
            <div className="modal-actions">
              <button className="ghost" onClick={() => setDeletingDoc(null)}>Cancel</button>
              <button className="danger" onClick={confirmDelete}>Delete permanently</button>
            </div>
          </div>
        </div>
      )}

      <main className="chat">
        <header className="bar">
          <button className="icon-btn menu" onClick={() => setDrawer(true)} aria-label="Open documents">☰</button>
          <div className="bar-title">
            <strong>Asking {scopeName}</strong>
          </div>
          <div className="seg" role="group" aria-label="Answer mode">
            <button className={mode === "standard" ? "on" : ""} onClick={() => setMode("standard")}
              title="Searches your documents and remembers the conversation">Standard</button>
            <button className={mode === "agentic" ? "on" : ""} onClick={() => setMode("agentic")}
              title="Decides where to look, checks the passages, and retries with a better query">Agentic</button>
          </div>
          <button className="ghost" onClick={newChat} disabled={!messages.length}>New chat</button>
        </header>

        {notice && (
          <div className={`notice ${notice.tone}`} role="status">
            <span>{notice.text}</span>
            <button className="icon-btn" onClick={() => setNotice(null)} aria-label="Dismiss">✕</button>
          </div>
        )}

        <section className="feed" aria-live="polite">
          {messages.length === 0 && (
            <div className="hello">
              <h2>Ask your documents anything</h2>
              <p>Upload a PDF, index it, then ask a question. Every answer shows the pages it came from.</p>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} m={m} />)}
          {thinking && <div className="msg bot"><span className="dots" aria-label="Thinking"><i /><i /><i /></span></div>}
          <div ref={endRef} />
        </section>

        <form className="composer" onSubmit={onSend}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) onSend(e); }}
            placeholder="Ask a question about your documents"
            rows={1}
            aria-label="Your question"
          />
          <button className="primary" type="submit" disabled={!input.trim() || thinking}>Ask</button>
        </form>
      </main>
    </div>
  );
}

function Message({ m }) {
  if (m.role === "user") return <div className="msg me">{m.text}</div>;
  if (m.error) return <div className="msg bot err">{m.error}</div>;

  // The same page can be cited by several passages; show each page once.
  const seen = new Set();
  const pages = (m.sources || []).filter((s) => {
    const key = `${s.filename}|${s.page_number}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return (
    <div className="msg bot">
      {m.route === "general" && <p className="meta">Answered from general knowledge, not your documents.</p>}
      {m.rewritten && <p className="meta">Searched again as: “{m.rewritten}”</p>}
      <div className="answer">{m.text}</div>
      {pages.length > 0 && (
        <ul className="pages" aria-label="Sources">
          {pages.map((s) => (
            <li key={`${s.filename}${s.page_number}`} title={s.filename}>
              <span className="pg">p.{s.page_number}</span>
              <span className="fn">{s.filename}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
