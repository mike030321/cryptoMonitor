import { drizzle } from "drizzle-orm/node-postgres";
import * as schema from "./schema";
import { MssqlPgPool } from "./mssql-pool";

// Use a fake pg-compatible pool backed by MSSQL.
// Drizzle generates PostgreSQL SQL; the pool translates it to T-SQL at
// execution time so all app code and schema files remain unchanged.
export const pool = new MssqlPgPool();
export const db = drizzle(pool as any, { schema });

export * from "./schema";
