# @workspace/api-client-react

React Query client + TypeScript types generated from `lib/api-spec/openapi.yaml`
via Orval.

## Source of truth

- OpenAPI spec: `lib/api-spec/openapi.yaml`
- Orval config: `lib/api-spec/orval.config.ts`
- Generated TypeScript source (committed): `src/generated/`
- Generated `.d.ts` declarations (gitignored): `dist/generated/`

## Why both `src/generated/` and `dist/generated/` exist

The package's `exports` field points at `src/index.ts`, so at runtime and at
bundle time consumers always see fresh source. However, this package is
declared with `composite: true` and is referenced as a TypeScript project
reference from consumer artifacts (e.g. `artifacts/crypto-monitor`). Project
references make `tsc -p tsconfig.json` resolve the package via its emitted
`.d.ts` files in `dist/`, not the `.ts` source. If `dist/` is missing or
stale relative to `src/`, consumer `tsc --noEmit` will fail with `TS6305`
("Output file ... has not been built from source file ...") or with
phantom missing-property errors when fields were added to the source after
the last build.

## Keeping `dist/` in sync

Two cases:

1. **After regenerating from the OpenAPI spec.** Run, from the repo root:

   ```sh
   pnpm --filter @workspace/api-spec run codegen
   pnpm --filter @workspace/api-client-react run build
   ```

   The first command rewrites `src/generated/`. The second rebuilds
   `dist/generated/` from it.

2. **After a fresh checkout / install.** `scripts/post-merge.sh` runs
   `pnpm --filter @workspace/api-client-react run build` automatically after
   `pnpm install --frozen-lockfile`, so a normal merge or fresh clone leaves
   `dist/` populated. If you ever skip post-merge, run the build command
   above by hand.

## Manual cleanup

```sh
pnpm --filter @workspace/api-client-react run clean
pnpm --filter @workspace/api-client-react run build
```
