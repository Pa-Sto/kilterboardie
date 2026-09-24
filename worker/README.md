# Kilterboardie Worker

Environment variables required:
- `DATA_GITHUB_TOKEN`: Fine-grained PAT with `contents:write` on the private repo.
- `DATA_REPO_OWNER`: Private dataset repo owner (e.g. `Pa-Sto`).
- `DATA_REPO_NAME`: Private dataset repo name (e.g. `kilterboardie-feedback`).
- `DATA_REPO_BRANCH`: Private dataset branch, e.g. `main`.
- `ALLOWED_ORIGINS`: Comma-separated browser origins allowed to submit feedback.

Endpoints:
- `POST /feedback` with `{ requestId, grade, angle, model, suggestedGrade, userFeedback, createdAt }` stores feedback in the private repo.

Generation is handled by the self-hosted API. The obsolete `PUBLIC_GITHUB_TOKEN`
Worker secret should be deleted after deploying this version.
