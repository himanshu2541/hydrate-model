// api.js -- thin fetch wrappers over the FastAPI backend.

async function _postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function _getJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function _putJson(url, body) {
  const res = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function _deleteJson(url) {
  const res = await fetch(url, { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

export const api = {
  getConfig: () => _getJson("/api/config"),
  run: (req) => _postJson("/api/run", req),
  sweepDiox: (req) => _postJson("/api/sweep/diox", req),
  sweepRatio: (req) => _postJson("/api/sweep/ratio", req),
  parseCustomExpData: (text) => _postJson("/api/exp-data/custom", { text }),
  cacheInfo: () => _getJson("/api/cache"),
  cacheClear: () => _postJson("/api/cache/clear", {}),
  listProjects: () => _getJson("/api/projects"),
  createProject: (name) => _postJson("/api/projects", { name }),
  getProject: (id) => _getJson(`/api/projects/${id}`),
  renameProject: (id, name) => _putJson(`/api/projects/${id}`, { name }),
  deleteProject: (id) => _deleteJson(`/api/projects/${id}`),
  saveProjectConfig: (id, config) => _putJson(`/api/projects/${id}/config`, { config }),
};
