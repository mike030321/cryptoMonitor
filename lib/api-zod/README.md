# @workspace/api-zod

Zod validation schemas + TypeScript types generated from
`lib/api-spec/openapi.yaml` via Orval.

## Source of truth

- OpenAPI spec: `lib/api-spec/openapi.yaml`
- Orval config: `lib/api-spec/orval.config.ts`
- Generated TypeScript source (committed): `src/generated/`
- Generated `.d.ts` declarations (gitignored): `dist/generated/`

## Why both `src/generated/` and `dist/generated/` exist

The package's `exports` field points at `src/index.ts`, so at runtime and at
bundle time consumers always see fresh source. However, this package is
declared with `composite: true` and is referenced as a TypeScript project
reference from the root `tsconfig.json`. Project references make
`tsc --build` resolve the package via its emitted `.d.ts` files in `dist/`,
not the `.ts` source. If `dist/` is missing or stale relative to `src/`,
`pnpm run typecheck:libs` will fail with `TS6305` ("Output file ... has not
been built from source file ...") or with phantom missing-property errors
when fields were added to the source after the last build.

## Re-export ambiguity

Orval generates the request-param zod schema (e.g. `GetCoinDetailParams` in
`src/generated/api.ts`) under the same name as the corresponding TS type
(`src/generated/types/getCoinDetailParams.ts`). Re-exporting both with
`export *` / `export type *` from `src/index.ts` would trigger TS2308
("has already exported a member named ..."). `src/index.ts` resolves the
ambiguity with an explicit re-export of the zod schemas. The matching TS
shape stays reachable via `z.infer<typeof GetCoinDetailParams>`. Keep the
explicit re-export in sync with any new colliding param names that appear
after running codegen.

## Keeping `dist/` in sync

Two cases:

1. **After regenerating from the OpenAPI spec.** Run, from the repo root:

   ```sh
   pnpm --filter @workspace/api-spec run codegen
   pnpm --filter @workspace/api-zod run build
   ```

   The first command rewrites `src/generated/`. The second rebuilds
   `dist/generated/` from it.

2. **After a fresh checkout / install.** `scripts/post-merge.sh` runs
   `pnpm --filter @workspace/api-zod run build` automatically after
   `pnpm install --frozen-lockfile`, so a normal merge or fresh clone leaves
   `dist/` populated. If you ever skip post-merge, run the build command
   above by hand.

## Manual cleanup

```sh
pnpm --filter @workspace/api-zod run clean
pnpm --filter @workspace/api-zod run build
```
