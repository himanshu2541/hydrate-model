// projects.js -- project bar: switch/create/rename/delete/save named
// bundles of the Setup page's input config (COMSOL "model file" analogue).
// Explicit Save button, no autosave -- avoids silently overwriting a
// project's saved config while the user is just experimenting.
import { api } from "./api.js";

const $ = (root, sel) => root.querySelector(sel);

export function initProjectBar(bar, modal, { getSetupState, applySetupState }, logEvent) {
  const select = $(bar, '[data-testid="project-select"]');
  const newBtn = $(bar, '[data-testid="project-new"]');
  const saveBtn = $(bar, '[data-testid="project-save"]');
  const renameBtn = $(bar, '[data-testid="project-rename"]');
  const deleteBtn = $(bar, '[data-testid="project-delete"]');

  const modalNameInput = $(modal, '[data-testid="project-modal-name"]');
  const modalCreateBtn = $(modal, '[data-testid="project-modal-create"]');
  const modalCancelBtn = $(modal, '[data-testid="project-modal-cancel"]');

  let projects = [];
  let activeId = null;

  function openModal() {
    modalNameInput.value = "";
    modal.hidden = false;
    modalNameInput.focus();
  }
  function closeModal() { modal.hidden = true; }

  function populateSelect() {
    select.innerHTML = "";
    for (const p of projects) {
      const o = document.createElement("option");
      o.value = String(p.id); o.textContent = p.name;
      select.appendChild(o);
    }
    if (activeId != null) select.value = String(activeId);
  }

  async function refreshList() {
    projects = await api.listProjects();
    if (!projects.length) {
      const created = await api.createProject("Default Project");
      projects = [created];
      logEvent("info", `Created "${created.name}" (no projects existed yet).`);
    }
    populateSelect();
  }

  async function selectProject(id, { applyConfig = true } = {}) {
    activeId = id;
    select.value = String(id);
    if (!applyConfig) return;
    const detail = await api.getProject(id);
    applySetupState(detail.config);
  }

  newBtn.addEventListener("click", openModal);
  modalCancelBtn.addEventListener("click", closeModal);
  modalCreateBtn.addEventListener("click", async () => {
    const name = modalNameInput.value.trim();
    if (!name) { alert("Project name cannot be empty."); return; }
    const created = await api.createProject(name);
    projects.push(created);
    populateSelect();
    await selectProject(created.id, { applyConfig: false });
    closeModal();
    logEvent("success", `Project "${created.name}" created.`);
  });

  select.addEventListener("change", async () => {
    const id = parseInt(select.value, 10);
    await selectProject(id);
    const p = projects.find((x) => x.id === id);
    logEvent("info", `Switched to project "${p ? p.name : id}".`);
  });

  saveBtn.addEventListener("click", async () => {
    if (activeId == null) return;
    let cfg;
    try { cfg = getSetupState(); } catch (exc) { alert(exc.message); return; }
    const updated = await api.saveProjectConfig(activeId, cfg);
    logEvent("success", `Saved current Setup into project "${updated.name}".`);
  });

  renameBtn.addEventListener("click", async () => {
    if (activeId == null) return;
    const current = projects.find((x) => x.id === activeId);
    const name = prompt("Rename project:", current ? current.name : "");
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) { alert("Project name cannot be empty."); return; }
    const updated = await api.renameProject(activeId, trimmed);
    const idx = projects.findIndex((x) => x.id === activeId);
    if (idx >= 0) projects[idx] = { ...projects[idx], name: updated.name };
    populateSelect();
    logEvent("info", `Project renamed to "${updated.name}".`);
  });

  deleteBtn.addEventListener("click", async () => {
    if (activeId == null) return;
    const current = projects.find((x) => x.id === activeId);
    if (!confirm(`Delete project "${current ? current.name : activeId}"?\nThis cannot be undone.`)) return;
    await api.deleteProject(activeId);
    logEvent("info", `Project "${current ? current.name : activeId}" deleted.`);
    await refreshList();
    if (projects.length) await selectProject(projects[0].id);
  });

  async function init() {
    await refreshList();
    if (projects.length) await selectProject(projects[0].id, { applyConfig: false });
  }

  return { init };
}
