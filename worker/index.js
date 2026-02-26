const JSON_HEADERS = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "POST,OPTIONS",
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

function parseGrade(value) {
  if (!value) return 6;
  const match = String(value).match(/V(\d+)/i);
  if (match) return Number(match[1]);
  const num = Number(value);
  return Number.isNaN(num) ? 6 : num;
}

async function githubRequest(url, token, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `token ${token}`,
      "Accept": "application/vnd.github+json",
      "User-Agent": "kilterboardie-worker",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GitHub API error ${res.status}: ${text}`);
  }
}

async function githubPutFile({ owner, repo, branch, token, path, contentBase64, message }) {
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${path}`;
  const res = await fetch(url, {
    method: "PUT",
    headers: {
      "Authorization": `token ${token}`,
      "Accept": "application/vnd.github+json",
      "User-Agent": "kilterboardie-worker",
    },
    body: JSON.stringify({
      message,
      content: contentBase64,
      branch,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GitHub API error ${res.status}: ${text}`);
  }
}

function encodeBase64Utf8(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function fetchAsBase64(url) {
  const res = await fetch(url, { method: "GET" });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function handleGenerate(request, env) {
  const payload = await request.json();
  const requestId = crypto.randomUUID();
  const grade = parseGrade(payload.grade);

  await githubRequest(
    `https://api.github.com/repos/${env.PUBLIC_REPO_OWNER}/${env.PUBLIC_REPO_NAME}/dispatches`,
    env.PUBLIC_GITHUB_TOKEN,
    {
      event_type: "generate-climb",
      client_payload: {
        grade,
        request_id: requestId,
      },
    }
  );

  return jsonResponse({ requestId });
}

async function handleFeedback(request, env) {
  const payload = await request.json();
  const requestId = payload.requestId || crypto.randomUUID();
  const now = new Date().toISOString();

  const entry = {
    request_id: requestId,
    grade: payload.grade,
    angle: payload.angle,
    model: payload.model,
    suggested_grade: payload.suggestedGrade,
    user_feedback: payload.userFeedback,
    matrix_path: payload.matrixPath,
    image_path: payload.imagePath,
    created_at: payload.createdAt,
    received_at: now,
  };

  const basePath = `feedback/${requestId}`;
  const message = `Add feedback ${requestId}`;

  await githubPutFile({
    owner: env.DATA_REPO_OWNER,
    repo: env.DATA_REPO_NAME,
    branch: env.DATA_REPO_BRANCH,
    token: env.DATA_GITHUB_TOKEN,
    path: `${basePath}/feedback.json`,
    contentBase64: encodeBase64Utf8(JSON.stringify(entry, null, 2)),
    message,
  });

  if (payload.matrixPath && env.PUBLIC_SITE_BASE) {
    const matrixUrl = `${env.PUBLIC_SITE_BASE.replace(/\/$/, "")}/${payload.matrixPath}`;
    const matrixBase64 = await fetchAsBase64(matrixUrl);
    await githubPutFile({
      owner: env.DATA_REPO_OWNER,
      repo: env.DATA_REPO_NAME,
      branch: env.DATA_REPO_BRANCH,
      token: env.DATA_GITHUB_TOKEN,
      path: `${basePath}/matrix.npy`,
      contentBase64: matrixBase64,
      message,
    });
  }

  if (payload.imagePath && env.PUBLIC_SITE_BASE) {
    const imageUrl = `${env.PUBLIC_SITE_BASE.replace(/\/$/, "")}/${payload.imagePath}`;
    const imageBase64 = await fetchAsBase64(imageUrl);
    await githubPutFile({
      owner: env.DATA_REPO_OWNER,
      repo: env.DATA_REPO_NAME,
      branch: env.DATA_REPO_BRANCH,
      token: env.DATA_GITHUB_TOKEN,
      path: `${basePath}/overlay.png`,
      contentBase64: imageBase64,
      message,
    });
  }

  return jsonResponse({ ok: true });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response("", { headers: JSON_HEADERS });
    }

    const url = new URL(request.url);
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405, headers: JSON_HEADERS });
    }

    if (url.pathname === "/generate") {
      return handleGenerate(request, env);
    }
    if (url.pathname === "/feedback") {
      return handleFeedback(request, env);
    }

    return new Response("Not Found", { status: 404, headers: JSON_HEADERS });
  },
};
