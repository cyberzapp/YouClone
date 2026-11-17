export type Model = "omni" | "zonos" | "qwen";
export type Settings = {
  model: Model; text: string; sample: string; language: string; style: string;
  temperature: number; speed: number; guidance: number; top_p: number;
  repetition_penalty: number; seed: number; polish: boolean; comma_pause: number; period_pause: number;
};
export type Voice = { id: string; name: string };
export type Sample = {
  id: string; voiceId: string; original: string; style: string; transcript: string;
  language: string; seconds: number; embedding?: number[] | null;
};
export type Project = { id: string; name: string };
export type Block = { id: string; projectId: string; settings: Settings; selected: string | null };
export type Take = { id: string; blockId: string; settings: Settings; seconds: number; created: number };
export type Library = { voices: Voice[]; samples: Sample[]; projects: Project[]; blocks: Block[]; takes: Take[]; sources: string[] };
export type Format = "wav" | "flac" | "mp3";
export type ExportSettings = { takes: string[]; format: Format; subtitles?: boolean; start?: number; end?: number; polish?: boolean; pause?: number };

declare global {
  interface Window {
    studio: {
      library(): Promise<Library>;
      service(): Promise<{ ready: boolean; serviceError?: string; modelWarning?: string | null; ws: string }>;
      project(project: Project, blocks: Block[]): Promise<void>;
      sample(name: string, sample: Omit<Sample, "voiceId">): Promise<{ voice: Voice; sample: Sample }>;
      take(take: Take): Promise<void>;
      export(settings: ExportSettings, quick: boolean, preview: boolean): Promise<{ url?: string; destination?: string }>;
    };
  }
}

export const defaultSettings: Settings = {
  model: "omni", text: "", sample: "", language: "", style: "", temperature: .7,
  speed: 1, guidance: 2, top_p: .95, repetition_penalty: 1.05, seed: 42, polish: true, comma_pause: .25, period_pause: .45,
};

export const describe = (error: unknown) => error instanceof Error ? error.message : String(error);
