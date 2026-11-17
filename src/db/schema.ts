import { index, integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const voices = sqliteTable("voices", {
  id: text().primaryKey(),
  name: text().notNull().unique(),
});

export const voiceSamples = sqliteTable("voice_samples", {
  id: text().primaryKey(),
  voiceId: text().notNull().references(() => voices.id),
  style: text().notNull(),
  original: text().notNull(),
  transcript: text().notNull(),
  language: text().notNull(),
  seconds: real().notNull(),
  embedding: text({ mode: "json" }).$type<number[]>(),
});

export const projects = sqliteTable("projects", {
  id: text().primaryKey(),
  name: text().notNull(),
  updated: integer().notNull(),
});

export const scriptBlocks = sqliteTable("script_blocks", {
  id: text().primaryKey(),
  projectId: text().notNull().references(() => projects.id, { onDelete: "cascade" }),
  position: integer().notNull(),
  settings: text({ mode: "json" }).$type<Record<string, unknown>>().notNull(),
  selected: text(),
}, table => [index("blocks_project").on(table.projectId, table.position)]);

export const audioTakes = sqliteTable("audio_takes", {
  id: text().primaryKey(),
  blockId: text().notNull().references(() => scriptBlocks.id, { onDelete: "cascade" }),
  seconds: real().notNull(),
  created: integer().notNull(),
  settings: text({ mode: "json" }).$type<Record<string, unknown>>().notNull(),
}, table => [index("takes_block").on(table.blockId, table.created)]);
