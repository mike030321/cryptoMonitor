#!/bin/bash
set -e
pnpm install --frozen-lockfile
pnpm --filter db push-force
# Rebuild generated API client declarations so consumers' typecheck
# (project references) doesn't trip over a stale or missing dist/.
# See lib/api-client-react/README.md and lib/api-zod/README.md for the
# codegen -> build flow.
pnpm --filter @workspace/api-client-react run build
pnpm --filter @workspace/api-zod run build
