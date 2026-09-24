const DEFAULT_ALLOWED_ORIGINS = ["https://pa-sto.github.io"];
const MAX_BODY_BYTES = 16 * 1024;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function allowedOrigins(env) {
  const configured = String(env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim().replace(/\/$/, ""))
    .filter(Boolean);
  return configured.length ? configured : DEFAULT_ALLOWED_ORIGINS;
}

function isAllowedOrigin(origin, env) {
  return Boolean(origin) && allowedOrigins(env).includes(origin.replace(/\/$/, ""));
}

function buildHeaders(origin) {
  return {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": "no-store",
    "Vary": "Origin",
    "X-Content-Type-Options": "nosniff",
  };
}

function jsonResponse(body, status, origin) {
  return new Response(JSON.stringify(body), { status, headers: buildHeaders(origin) });
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
    console.error(`GitHub API error ${res.status}: ${text.slice(0, 500)}`);
    throw new Error("Dataset storage failed");
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

async function readJson(request) {
  const contentLength = Number(request.headers.get("Content-Length") || 0);
  if (contentLength > MAX_BODY_BYTES) {
    throw new Error("PAYLOAD_TOO_LARGE");
  }
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) {
    throw new Error("PAYLOAD_TOO_LARGE");
  }
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error("INVALID_JSON");
  }
}

async function handleFeedback(request, env) {
  const payload = await readJson(request);
  const requestId = String(payload.requestId || "");
  if (!UUID_PATTERN.test(requestId)) {
    return { error: "Invalid request ID", status: 400 };
  }

  const userFeedback = String(payload.userFeedback || "").trim().slice(0, 2000);
  const suggestedGrade = String(payload.suggestedGrade || "").trim().slice(0, 32);
  if (!userFeedback && !suggestedGrade) {
    return { error: "Feedback is empty", status: 400 };
  }
  const now = new Date().toISOString();

  const entry = {
    request_id: requestId,
    grade: String(payload.grade || "").slice(0, 16),
    angle: String(payload.angle || "").slice(0, 16),
    model: String(payload.model || "").slice(0, 32),
    suggested_grade: suggestedGrade,
    user_feedback: userFeedback,
    created_at: String(payload.createdAt || "").slice(0, 64),
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

  return { body: { ok: true }, status: 200 };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return jsonResponse({ ok: true }, 200, origin || DEFAULT_ALLOWED_ORIGINS[0]);
    }

    if (!isAllowedOrigin(origin, env)) {
      return jsonResponse({ ok: false, error: "Origin not allowed" }, 403, DEFAULT_ALLOWED_ORIGINS[0]);
    }
    if (request.method === "OPTIONS") {
      return new Response("", { headers: buildHeaders(origin) });
    }

    try {
      if (request.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405, headers: buildHeaders(origin) });
      }

      if (url.pathname === "/feedback") {
        const result = await handleFeedback(request, env);
        if (result.error) {
          return jsonResponse({ ok: false, error: result.error }, result.status, origin);
        }
        return jsonResponse(result.body, result.status, origin);
      }

      return new Response("Not Found", { status: 404, headers: buildHeaders(origin) });
    } catch (error) {
      if (error instanceof Error && error.message === "PAYLOAD_TOO_LARGE") {
        return jsonResponse({ ok: false, error: "Payload too large" }, 413, origin);
      }
      if (error instanceof Error && error.message === "INVALID_JSON") {
        return jsonResponse({ ok: false, error: "Invalid JSON" }, 400, origin);
      }
      console.error(error);
      return jsonResponse({ ok: false, error: "Feedback could not be stored" }, 500, origin);
    }
  },
};
