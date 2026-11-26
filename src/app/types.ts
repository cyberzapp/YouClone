export type Model = "orpheus_seedvc";
export type EffectPreset = "broadcast" | "podcast" | "cinematic" | "warm" | "bright" | "raw";
export type PauseFill = "silence" | "room_tone";
export type FilterMode = "none" | "phone" | "radio" | "robot" | "megaphone";
export type SpaceMode = "none" | "room" | "hall" | "cathedral";
export type DistanceMode = "near" | "medium" | "far";
export type AmbienceMode = "none" | "vinyl" | "rain" | "tape" | "cafe";

export type VoiceCategory = "mysterious" | "cartoon" | "male" | "female";

export type VoiceDesignCandidate = {
  id: string;
  label: string;
  seconds: number;
  seed: number;
  url: string;
};

export type StudioVoice = {
  id: string;
  name: string;
  mode: "instant" | "professional" | "designed";
  category?: string;
  design_prompt?: string;
  is_default?: boolean;
  orpheus_adapter?: string | null;
  orpheus_voice?: string;
  seedvc_checkpoint?: string | null;
  seedvc_config?: string | null;
  ref_audio?: string | null;
  preview_file?: string | null;
  target_f0_hz?: number;
  sample_minutes?: number;
  diffusion_steps?: number;
  cfg_rate?: number;
  created: number;
  description?: string;
};

export type PvcTrainingState = {
  running: boolean;
  stage: string;
  percent: number;
  message: string;
  error: string | null;
  voice: StudioVoice | null;
};

export type Settings = {
  model: Model;
  text: string;
  studio_voice_id?: string;
  pronunciation?: string;
  timing?: { start: number; end: number };
  sample: string;
  language: string;
  style: string;
  temperature: number;
  speed: number;
  guidance: number;
  top_p: number;
  repetition_penalty: number;
  seed: number;
  polish: boolean;
  comma_pause: number;
  period_pause: number;
  num_steps?: number;
  class_temperature?: number;
  position_temperature?: number;
  lora_adapter?: string;
  stability?: number;
  similarity_boost?: number;
  style_exaggeration?: number;
  speaker_boost?: boolean;
  effect_preset?: EffectPreset;
  pause_fill?: PauseFill;
  trim_silence?: boolean;
  filter_mode?: FilterMode;
  space_mode?: SpaceMode;
  distance_mode?: DistanceMode;
  ambience_mode?: AmbienceMode;
  watermark?: boolean;
};

export type LoraAdapter = {
  id: string;
  name: string;
  rank: number;
  alpha: number;
  steps: number;
  clips: number;
  sources: string[];
  final_loss: number;
  sample_id: string;
  sample_transcript: string;
  sample_language: string;
  sample_seconds: number;
  created: number;
};

export type ModelStatusInfo = {
  key: Model;
  repo: string;
  label: string;
  sample_rate: number;
  cloning: boolean;
  cached: boolean;
  loaded: boolean;
};

export type Voice = { id: string; name: string };
export type Sample = {
  id: string;
  voiceId: string;
  original: string;
  style: string;
  transcript: string;
  language: string;
  seconds: number;
  embedding?: number[] | null;
};
export type Project = { id: string; name: string };
export type Block = { id: string; projectId: string; settings: Settings; selected: string | null };
export type Take = { id: string; blockId: string; settings: Settings; seconds: number; created: number };
export type Library = {
  voices: Voice[];
  samples: Sample[];
  projects: Project[];
  blocks: Block[];
  takes: Take[];
  sources: string[];
};
export type Format = "wav" | "flac" | "mp3" | "m4a";
export type ExportSettings = {
  takes: string[];
  format: Format;
  timing?: { start: number; end: number }[];
  subtitles?: boolean;
  start?: number;
  end?: number;
  polish?: boolean;
  pause?: number;
  effect_preset?: EffectPreset;
  pause_fill?: PauseFill;
};

declare global {
  interface Window {
    studio: {
      library(): Promise<Library>;
      service(): Promise<{ ready: boolean; serviceError?: string; modelWarning?: string | null; ws: string }>;
      project(project: Project, blocks: Block[]): Promise<void>;
      sample(name: string, sample: Omit<Sample, "voiceId">): Promise<{ voice: Voice; sample: Sample }>;
      take(take: Take): Promise<void>;
      export(
        settings: ExportSettings,
        quick: boolean,
        preview: boolean,
      ): Promise<{ url?: string; destination?: string }>;
    };
  }
}

export const defaultSettings: Settings = {
  model: "orpheus_seedvc",
  studio_voice_id: "",
  text: "",
  sample: "",
  language: "",
  style: "",
  temperature: 0.58,
  speed: 1,
  guidance: 2.0,
  top_p: 0.86,
  repetition_penalty: 1.18,
  seed: 202,
  polish: true,
  comma_pause: 0.22,
  period_pause: 0.34,
  num_steps: 35,
  class_temperature: 0,
  position_temperature: 5,
  lora_adapter: "",
  stability: 55,
  similarity_boost: 75,
  style_exaggeration: 45,
  speaker_boost: true,
  effect_preset: "broadcast",
  pause_fill: "silence",
  trim_silence: true,
  filter_mode: "none",
  space_mode: "none",
  distance_mode: "near",
  ambience_mode: "none",
  watermark: false,
};

export const describe = (error: unknown) => (error instanceof Error ? error.message : String(error));
