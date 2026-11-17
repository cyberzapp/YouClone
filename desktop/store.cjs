const Database = require("better-sqlite3");
const { drizzle } = require("drizzle-orm/better-sqlite3");
const { asc, eq, notInArray, and } = require("drizzle-orm");
const { randomUUID } = require("node:crypto");
const schema = require("./db/schema.js");

module.exports = function openStore(file) {
  const sqlite = new Database(file);
  sqlite.pragma("journal_mode = WAL");
  sqlite.pragma("foreign_keys = ON");
  const db = drizzle(sqlite);
  const { voices, voiceSamples, projects, scriptBlocks, audioTakes } = schema;
  sqlite.transaction(() => {
    if (sqlite.pragma("user_version", { simple: true }) >= 1) return;
    sqlite.exec(`
      CREATE TABLE voices (id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL UNIQUE);
      CREATE TABLE voice_samples (
        id TEXT PRIMARY KEY NOT NULL, voiceId TEXT NOT NULL REFERENCES voices(id),
        style TEXT NOT NULL, original TEXT NOT NULL, transcript TEXT NOT NULL,
        language TEXT NOT NULL, seconds REAL NOT NULL, embedding TEXT
      );
      CREATE TABLE projects (id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL, updated INTEGER NOT NULL);
      CREATE TABLE script_blocks (
        id TEXT PRIMARY KEY NOT NULL, projectId TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        position INTEGER NOT NULL, settings TEXT NOT NULL, selected TEXT
      );
      CREATE INDEX blocks_project ON script_blocks(projectId, position);
      CREATE TABLE audio_takes (
        id TEXT PRIMARY KEY NOT NULL, blockId TEXT NOT NULL REFERENCES script_blocks(id) ON DELETE CASCADE,
        seconds REAL NOT NULL, created INTEGER NOT NULL, settings TEXT NOT NULL
      );
      CREATE INDEX takes_block ON audio_takes(blockId, created);
      PRAGMA user_version = 1;
    `);
  })();
  return {
    read() {
      return {
        voices: db.select().from(voices).orderBy(asc(voices.name)).all(),
        samples: db.select().from(voiceSamples).all(),
        projects: db.select().from(projects).orderBy(asc(projects.updated)).all(),
        blocks: db.select().from(scriptBlocks).orderBy(asc(scriptBlocks.position)).all(),
        takes: db.select().from(audioTakes).orderBy(asc(audioTakes.created)).all(),
      };
    },
    saveProject(project, blocks) {
      db.transaction(tx => {
        const values = { id: project.id, name: project.name, updated: Date.now() };
        tx.insert(projects).values(values).onConflictDoUpdate({ target: projects.id, set: values }).run();
        const own = eq(scriptBlocks.projectId, project.id);
        tx.delete(scriptBlocks).where(blocks.length ? and(own, notInArray(scriptBlocks.id, blocks.map(b => b.id))) : own).run();
        for (const [position, { id, settings, selected }] of blocks.entries()) {
          const row = { id, projectId: project.id, position, settings, selected };
          tx.insert(scriptBlocks).values(row).onConflictDoUpdate({ target: scriptBlocks.id, set: row }).run();
        }
      });
    },
    saveSample(name, sample) {
      return db.transaction(tx => {
        const existing = tx.select().from(voices).where(eq(voices.name, name)).get();
        const voice = existing ?? { id: randomUUID(), name };
        if (!existing) tx.insert(voices).values(voice).run();
        const row = { ...sample, voiceId: voice.id };
        tx.insert(voiceSamples).values(row).run();
        return { voice, sample: row };
      });
    },
    saveTake(take) {
      db.transaction(tx => {
        tx.insert(audioTakes).values(take).run();
        tx.update(scriptBlocks).set({ selected: take.id }).where(eq(scriptBlocks.id, take.blockId)).run();
      });
    },
    close: () => sqlite.close(),
  };
};
