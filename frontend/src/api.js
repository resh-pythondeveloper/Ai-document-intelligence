const BASE = import.meta.env.VITE_API_URL ?? "";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Can't reach the server. Check that the backend is running.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const json = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const listDocuments = () => request("/api/v1/documents");

export const uploadDocument = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/api/v1/documents/upload", { method: "POST", body: form });
};

export const indexDocument = (id) =>
  request(`/api/v1/documents/${id}/ingest`, { method: "POST" });

export const askStandard = ({ question, documentId, sessionId }) =>
  request("/api/v1/documents/chat", json({
    question, document_id: documentId, session_id: sessionId, top_k: 5,
  }));

export const askAgentic = ({ question, documentId }) =>
  request("/api/v1/agentic-rag/", json({ question, document_id: documentId }));

export const deleteDocument = (id) =>
  request(`/api/v1/documents/${id}`, { method: "DELETE" });
