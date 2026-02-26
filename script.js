const gradeSelect = document.getElementById("gradeSelect");
const angleSelect = document.getElementById("angleSelect");
const modelSelect = document.getElementById("modelSelect");
const generateBtn = document.getElementById("generateBtn");
const generatedList = document.getElementById("generatedList");
const outputStatus = document.getElementById("outputStatus");

const API_BASE = window.KILTERBOARDIE_API || "";
const SITE_BASE = window.KILTERBOARDIE_SITE_BASE || "";
const cellCount = 36;
const climbCount = 1;
const STORAGE_KEY = "kilterboardieFeedback";
const datasetEntries = loadDataset();
let latestMeta = null;
let currentRequestId = null;
let pollTimer = null;

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

function assetUrl(path) {
  if (!SITE_BASE) {
    return path;
  }
  return `${SITE_BASE.replace(/\\/$/, "")}/${path}`;
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
  const meta = data?.meta;

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
  metaId.textContent = meta?.request_id ? `Run ${meta.request_id}` : `Sample ${index + 1}`;

  meta.append(metaGrade, metaAngle, metaModel, metaId);

  const detail = document.createElement("p");
  detail.textContent = meta?.created_at
    ? `Generated on ${new Date(meta.created_at).toLocaleString()}.`
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

    if (API_BASE && meta?.request_id) {
      sendButton.disabled = true;
      feedbackStatus.textContent = "Sending...";

      fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requestId: meta.request_id,
          grade: gradeSelect.value,
          angle: angleSelect.value,
          model: modelSelect.value,
          suggestedGrade,
          userFeedback,
          matrixPath: meta.matrix_path,
          imagePath: meta.image_path,
          createdAt: meta.created_at,
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
  if (!API_BASE) {
    renderLocal();
    return;
  }

  outputStatus.textContent = "Queued...";
  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grade: gradeSelect.value,
        angle: angleSelect.value,
        model: modelSelect.value,
      }),
    });
    if (!res.ok) {
      throw new Error("Generation failed");
    }
    const data = await res.json();
    currentRequestId = data.requestId;
    if (pollTimer) {
      window.clearTimeout(pollTimer);
    }
    pollForResult(currentRequestId);
  } catch (error) {
    outputStatus.textContent = "Generation failed.";
    renderLocal();
  }
}

function renderClimbs() {
  generatedList.innerHTML = "";
  outputStatus.textContent = "Generating...";
  latestMeta = null;
  currentRequestId = null;
  requestGeneration();
}

generateBtn.addEventListener("click", renderClimbs);

renderClimbs();
