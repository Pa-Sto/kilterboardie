# Kilterboardie Worker

Environment variables required:
- `PUBLIC_GITHUB_TOKEN`: Fine-grained PAT with Actions write on the public repo.
- `DATA_GITHUB_TOKEN`: Fine-grained PAT with `contents:write` on the private repo.
- `PUBLIC_REPO_OWNER`: Public site repo owner.
- `PUBLIC_REPO_NAME`: Public site repo name.
- `DATA_REPO_OWNER`: Private dataset repo owner (e.g. `Pa-Sto`).
- `DATA_REPO_NAME`: Private dataset repo name (e.g. `kilterboardie-feedback`).
- `DATA_REPO_BRANCH`: Private dataset branch, e.g. `main`.
- `PUBLIC_SITE_BASE`: Base URL for GitHub Pages, e.g. `https://pa-sto.github.io/kilterboardie`.

Endpoints:
- `POST /generate` with `{ grade, angle, model }` triggers GitHub Actions.
- `POST /feedback` with `{ requestId, grade, angle, model, suggestedGrade, userFeedback, matrixPath, imagePath, createdAt }` stores feedback in the private repo.
