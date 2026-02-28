const gradeSelect = document.getElementById("gradeSelect");
const angleSelect = document.getElementById("angleSelect");
const modelSelect = document.getElementById("modelSelect");
const generateBtn = document.getElementById("generateBtn");
const generatedList = document.getElementById("generatedList");
const outputStatus = document.getElementById("outputStatus");
const outputSpinner = document.getElementById("outputSpinner");

const urlParams = new URLSearchParams(window.location.search);
const apiOverride = urlParams.get("api");
const siteOverride = urlParams.get("site");
const isLocalContext =
  window.location.protocol === "file:" ||
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const defaultLocalApi = isLocalContext ? "http://127.0.0.1:8000" : "";
const API_BASE = normalizeBase(apiOverride || window.KILTERBOARDIE_API || defaultLocalApi);
const API_FALLBACK = normalizeBase(window.KILTERBOARDIE_API_FALLBACK || "");
const API_HEALTH_PATH = window.KILTERBOARDIE_API_HEALTH || "/health";
const SITE_BASE = normalizeBase(siteOverride || window.KILTERBOARDIE_SITE_BASE || "");
const cellCount = 36;
const climbCount = 1;
const STORAGE_KEY = "kilterboardieFeedback";
const datasetEntries = loadDataset();
let latestMeta = null;
let currentRequestId = null;
let pollTimer = null;
let activeApi = API_BASE;
let apiReadyPromise = null;

function loadDataset() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (error) {
    return [];
  }
}

function persistDataset() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(datasetEntries));
  } catch (error) {
    // Ignore storage failures (private browsing, quota limits).
  }
}

function normalizeBase(base) {
  if (!base) {
    return "";
  }
  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function assetUrl(path) {
  if (!SITE_BASE) {
    return path;
  }
  return `${SITE_BASE}/${path}`;
}

async function checkApiHealth(base) {
  if (!base) {
    return false;
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 1500);
  try {
    const res = await fetch(`${base}${API_HEALTH_PATH}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    return res.ok;
  } catch (error) {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function ensureApi() {
  if (apiReadyPromise) {
    return apiReadyPromise;
  }
  apiReadyPromise = (async () => {
    if (await checkApiHealth(API_BASE)) {
      activeApi = API_BASE;
      return activeApi;
    }
    if (API_FALLBACK && (await checkApiHealth(API_FALLBACK))) {
      activeApi = API_FALLBACK;
      outputStatus.textContent = "Primary API offline, using fallback (slow).";
      return activeApi;
    }
    activeApi = "";
    return "";
  })();
  return apiReadyPromise;
}

function createCell(active) {
  const cell = document.createElement("div");
  cell.className = active ? "grid-cell active" : "grid-cell";
  return cell;
}

function createGridPreview(matrix) {
  const grid = document.createElement("div");
  grid.className = "grid-preview";
  for (let i = 0; i < cellCount; i += 1) {
    grid.appendChild(createCell(matrix[i]));
  }
  return grid;
}

function setSpinner(isActive) {
  if (!outputSpinner) {
    return;
  }
  outputSpinner.classList.toggle("is-active", isActive);
}

function createImagePreview(url) {
  const img = document.createElement("img");
  img.className = "climb-image";
  img.alt = "Generated climb overlay";
  img.loading = "lazy";
  img.src = url;
  return img;
}

function buildCard(index, data) {
  const matrix = data?.matrix ?? Array.from({ length: cellCount }, () => Math.random() > 0.72);
  const imageUrl = data?.imageUrl;
  const metaInfo = data?.meta;

  const card = document.createElement("div");
  card.className = "climb-card";

  const meta = document.createElement("div");
  meta.className = "climb-meta";

  const metaGrade = document.createElement("span");
  metaGrade.textContent = `Grade: ${gradeSelect.value}`;

  const metaAngle = document.createElement("span");
  metaAngle.textContent = `Angle: ${angleSelect.value}°`;

  const metaModel = document.createElement("span");
  metaModel.textContent = `Model: ${modelSelect.value}`;

  const metaId = document.createElement("span");
  metaId.textContent = metaInfo?.request_id ? `Run ${metaInfo.request_id}` : `Sample ${index + 1}`;

  meta.append(metaGrade, metaAngle, metaModel, metaId);

  const detail = document.createElement("p");
  detail.textContent = metaInfo?.created_at
    ? `Generated on ${new Date(metaInfo.created_at).toLocaleString()}.`
    : "Generated layout preview (synthetic placeholder).";

  const feedback = document.createElement("div");
  feedback.className = "card-feedback";

  const suggestedLabel = document.createElement("label");
  suggestedLabel.className = "field";
  const suggestedText = document.createElement("span");
  suggestedText.className = "field-label";
  suggestedText.textContent = "Suggested Grade";
  const suggestedInput = document.createElement("input");
  suggestedInput.className = "field-control";
  suggestedInput.type = "text";
  suggestedInput.placeholder = "e.g., V6 / 6C";
  suggestedInput.name = `suggested-grade-${index}`;
  suggestedLabel.append(suggestedText, suggestedInput);

  const feedbackLabel = document.createElement("label");
  feedbackLabel.className = "field";
  const feedbackText = document.createElement("span");
  feedbackText.className = "field-label";
  feedbackText.textContent = "User Feedback";
  const feedbackInput = document.createElement("input");
  feedbackInput.className = "field-control";
  feedbackInput.type = "text";
  feedbackInput.placeholder = "How did it feel?";
  feedbackInput.name = `user-feedback-${index}`;
  feedbackLabel.append(feedbackText, feedbackInput);

  feedback.append(suggestedLabel, feedbackLabel);

  const feedbackActions = document.createElement("div");
  feedbackActions.className = "feedback-actions";

  const sendButton = document.createElement("button");
  sendButton.className = "secondary";
  sendButton.type = "button";
  sendButton.textContent = "Send Feedback";

  const feedbackStatus = document.createElement("span");
  feedbackStatus.className = "feedback-status";

  sendButton.addEventListener("click", () => {
    const suggestedGrade = suggestedInput.value.trim();
    const userFeedback = feedbackInput.value.trim();

    if (API_BASE && metaInfo?.request_id) {
      sendButton.disabled = true;
      feedbackStatus.textContent = "Sending...";

      fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requestId: metaInfo.request_id,
          grade: gradeSelect.value,
          angle: angleSelect.value,
          model: modelSelect.value,
          suggestedGrade,
          userFeedback,
          matrixPath: metaInfo.matrix_path,
          imagePath: metaInfo.image_path,
          createdAt: metaInfo.created_at,
        }),
      })
        .then((res) => {
          if (!res.ok) {
            throw new Error("Feedback failed");
          }
          feedbackStatus.textContent = "Feedback saved.";
        })
        .catch(() => {
          feedbackStatus.textContent = "Could not send feedback.";
        })
        .finally(() => {
          sendButton.disabled = false;
        });
      return;
    }

    const entry = {
      id: `sample-${Date.now()}-${index}`,
      grade: gradeSelect.value,
      angle: angleSelect.value,
      model: modelSelect.value,
      matrix,
      suggestedGrade,
      userFeedback,
      createdAt: new Date().toISOString(),
    };

    datasetEntries.push(entry);
    persistDataset();
    feedbackStatus.textContent = "Feedback saved.";
  });

  feedbackActions.append(sendButton, feedbackStatus);

  const preview = imageUrl ? createImagePreview(imageUrl) : createGridPreview(matrix);
  card.append(meta, detail, preview, feedback, feedbackActions);
  return card;
}

function renderLocal() {
  generatedList.innerHTML = "";
  for (let i = 0; i < climbCount; i += 1) {
    generatedList.appendChild(buildCard(i));
  }
  setSpinner(false);
  outputStatus.textContent = "Updated";
}

async function pollForResult(requestId) {
  const metaUrl = assetUrl(`generated/latest.json?ts=${Date.now()}`);
  try {
    const res = await fetch(metaUrl, { cache: "no-store" });
    if (res.ok) {
      const meta = await res.json();
      if (meta.request_id === requestId) {
        latestMeta = meta;
        generatedList.innerHTML = "";
        generatedList.appendChild(
          buildCard(0, {
            imageUrl: assetUrl(`generated/latest.png?ts=${Date.now()}`),
            meta,
          })
        );
        setSpinner(false);
        outputStatus.textContent = "Updated";
        return;
      }
    }
  } catch (error) {
    // Ignore polling errors and retry.
  }

  pollTimer = window.setTimeout(() => pollForResult(requestId), 2500);
}

async function requestGeneration() {
  const apiBase = await ensureApi();
  if (!apiBase) {
    renderLocal();
    return;
  }

  outputStatus.textContent = "Queued...";
  setSpinner(true);
  try {
    const res = await fetch(`${apiBase}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grade: gradeSelect.value,
        angle: angleSelect.value,
        model: modelSelect.value,
      }),
    });
    if (!res.ok) {
      let message = "Generation failed.";
      try {
        const err = await res.json();
        message = err?.error ? `Generation failed: ${err.error}` : message;
      } catch (error) {
        const text = await res.text();
        if (text) {
          message = `Generation failed: ${text}`;
        }
      }
      outputStatus.textContent = message;
      setSpinner(false);
      return;
    }
    const data = await res.json();
    if (data?.imageDataUrl) {
      latestMeta = data.meta || null;
      generatedList.innerHTML = "";
      generatedList.appendChild(
        buildCard(0, {
          imageUrl: data.imageDataUrl,
          meta: data.meta,
        })
      );
      setSpinner(false);
      outputStatus.textContent = "Updated";
      return;
    }

    currentRequestId = data.requestId;
    if (pollTimer) {
      window.clearTimeout(pollTimer);
    }
    pollForResult(currentRequestId);
  } catch (error) {
    outputStatus.textContent = "Generation failed.";
    setSpinner(false);
    renderLocal();
  }
}

function renderClimbs() {
  generatedList.innerHTML = "";
  outputStatus.textContent = "Generating...";
  setSpinner(true);
  latestMeta = null;
  currentRequestId = null;
  requestGeneration();
}

generateBtn.addEventListener("click", renderClimbs);

renderClimbs();
