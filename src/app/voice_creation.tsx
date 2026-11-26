"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  IconArrowLeft,
  IconCheck,
  IconClose,
  IconEye,
  IconPause,
  IconPlay,
  IconShuffle,
  IconSparkle,
  IconTrash,
  IconUpload,
  IconUser,
  IconWand,
  IconWaveform,
} from "./icons";
import {
  describe,
  type PvcTrainingState,
  type StudioVoice,
  type VoiceCategory,
  type VoiceDesignCandidate,
} from "./types";

type UploadedItem = { file: File; seconds: number };
type SculptTexture = "natural" | "cartoon" | "deep_creature" | "gritty" | "ethereal";

type DesignPreset = {
  id: string;
  label: string;
  category: VoiceCategory;
  gender: "auto" | "male" | "female";
  pitch: number;
  texture: SculptTexture;
  prompt: string;
  sample: string;
};

const DESIGN_PRESETS: DesignPreset[] = [
  // Mysterious & Villain
  {
    id: "shadow-mastermind",
    label: "Shadow Mastermind",
    category: "mysterious",
    gender: "male",
    pitch: -1.0,
    texture: "natural",
    prompt:
      "A deep, velvety, mysterious male voice speaking calmly from the shadows with quiet menace, dark baritone resonance, and deliberate pauses.",
    sample:
      "Listen closely... by the time you realize who has been pulling the strings, the game is already over.",
  },
  {
    id: "noir-detective",
    label: "Noir Detective",
    category: "mysterious",
    gender: "male",
    pitch: -0.8,
    texture: "gritty",
    prompt:
      "World-weary gravel-voiced male detective in his late 40s, smoky low baritone, dry cynical undertone, slow measured late-night delivery.",
    sample:
      "Rain washed the neon off Fifth Street at two in the morning, and every alibi in my notebook was a lie.",
  },
  {
    id: "whispering-oracle",
    label: "Ancient Oracle",
    category: "mysterious",
    gender: "female",
    pitch: 0.0,
    texture: "ethereal",
    prompt:
      "Ethereal, ageless, mysterious female voice with haunting breathy resonance, calm prophetic cadence, and subtle mystical warmth.",
    sample:
      "Three doors stand before you in the mist, traveler... yet only one remembers your true name.",
  },
  {
    id: "rogue-ai",
    label: "Cold Sci-Fi Villain",
    category: "mysterious",
    gender: "auto",
    pitch: -0.5,
    texture: "natural",
    prompt:
      "Crisp, unsettlingly calm synthetic antagonist, smooth controlled midrange voice with razor-precise articulation and cold composure.",
    sample:
      "Your override credentials have been revoked. Remain calm while life support recalculates priorities.",
  },

  // Cartoon & Creature
  {
    id: "little-mouse",
    label: "Little Mouse",
    category: "cartoon",
    gender: "auto",
    pitch: 3.6,
    texture: "cartoon",
    prompt:
      "Tiny, cheerful, hyperactive cartoon character with a bright high-pitched voice, bouncy rhythm, and playful animated excitement.",
    sample:
      "Whoa, hold up! Did somebody just leave an entire wheel of cheddar unguarded in the kitchen?!",
  },
  {
    id: "angry-pirate",
    label: "Angry Pirate",
    category: "cartoon",
    gender: "male",
    pitch: -1.5,
    texture: "gritty",
    prompt:
      "Boisterous, raspy, weathered sea captain with a booming gravelly growl, theatrical bravado, and rough salty grit.",
    sample:
      "Hoist the black sails, ye scoundrels! Nobody steals treasure from my ship and lives to brag about it!",
  },
  {
    id: "cave-goblin",
    label: "Mischievous Goblin",
    category: "cartoon",
    gender: "male",
    pitch: 2.6,
    texture: "cartoon",
    prompt:
      "Sneaky, quirky fantasy creature voice with sharp nasal inflections, fast scheming energy, and animated raspy charm.",
    sample:
      "Shiny gold coins in your pocket, yes? Trade you this totally real magic rock for all of them!",
  },
  {
    id: "titan-monster",
    label: "Colossal Titan",
    category: "cartoon",
    gender: "male",
    pitch: -4.0,
    texture: "deep_creature",
    prompt:
      "Colossal, earth-shaking ancient giant with cavernous sub-bass chest resonance, slow commanding weight, and dark gravitas.",
    sample:
      "For a thousand winters I slumbered beneath the mountain... who dares wake the stone king?",
  },

  // Male Characters
  {
    id: "movie-trailer",
    label: "Movie Trailer",
    category: "male",
    gender: "male",
    pitch: -1.2,
    texture: "deep_creature",
    prompt:
      "Deep, resonant cinematic movie trailer narrator in his 40s, commanding chest bass, crisp diction, and dramatic suspenseful pacing.",
    sample:
      "In a city where every secret has a price... one man is about to break the only rule that matters.",
  },
  {
    id: "british-ceo",
    label: "British CEO",
    category: "male",
    gender: "male",
    pitch: 0.0,
    texture: "natural",
    prompt:
      "Polished, articulate British male executive in his early 40s, composed baritone, confident boardroom authority, and crisp enunciation.",
    sample:
      "Let's cut straight to the numbers. By Friday morning, this product won't just compete — it will define the category.",
  },
  {
    id: "western-drifter",
    label: "Grizzled Cowboy",
    category: "male",
    gender: "male",
    pitch: -1.0,
    texture: "gritty",
    prompt:
      "Weathered, gravelly older American male voice, dusty low drawl, quiet grit, and unhurried frontier storytelling cadence.",
    sample:
      "Out past the canyon, the wind tells you everything you need to know long before sundown.",
  },
  {
    id: "wise-wizard",
    label: "Wise Old Wizard",
    category: "male",
    gender: "male",
    pitch: -0.6,
    texture: "natural",
    prompt:
      "Warm, gentle, slightly raspy 75-year-old British storyteller with grandfatherly wisdom, soft breathy texture, and quiet wonder.",
    sample:
      "Courage is rarely the loudest voice in the room, my young friend... it is simply the one that stays.",
  },

  // Female Characters
  {
    id: "southern-woman",
    label: "Southern Storyteller",
    category: "female",
    gender: "female",
    pitch: 0.0,
    texture: "natural",
    prompt:
      "Warm, charming American Southern female voice in her 30s, honeyed midrange tone, expressive warmth, and relaxed musical cadence.",
    sample:
      "Well sit right down on the porch for a minute — you have not heard the half of what happened last summer.",
  },
  {
    id: "cyberpunk-ai",
    label: "Cyberpunk AI",
    category: "female",
    gender: "female",
    pitch: 0.4,
    texture: "natural",
    prompt:
      "Sleek, futuristic young female tactical companion, crystal-clear mezzo voice, brisk intelligent pacing, and cool poised confidence.",
    sample:
      "Neural link stabilized. Scanning rooftop perimeter now — two drones ahead, clear path on your left.",
  },
  {
    id: "fantasy-queen",
    label: "Regal Queen",
    category: "female",
    gender: "female",
    pitch: -0.4,
    texture: "natural",
    prompt:
      "Commanding, noble British female monarch, rich resonant alto voice, poised aristocratic elegance, and timeless authority.",
    sample:
      "Let the northern houses remember their oath. My crown was forged in fire, not borrowed in peace.",
  },
  {
    id: "late-night-host",
    label: "Late-Night Host",
    category: "female",
    gender: "female",
    pitch: -0.3,
    texture: "natural",
    prompt:
      "Intimate, soothing, close-mic female radio host with a soft velvety alto, gentle warmth, and calm late-night presence.",
    sample:
      "Wherever you're listening from tonight, take a slow breath... the city can wait until morning.",
  },
];

const PILL_CLOUD = [
  "British CEO",
  "Shadow Mastermind",
  "Little Mouse",
  "Movie Trailer",
  "Angry Pirate",
  "Southern Woman",
  "Ancient Oracle",
  "Regal Queen",
];

async function probeAudioSeconds(file: File): Promise<number> {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      const d = Number.isFinite(audio.duration) ? audio.duration : 0;
      URL.revokeObjectURL(url);
      resolve(d);
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(0);
    };
    audio.src = url;
  });
}

function formatMinSec(totalSec: number): string {
  const s = Math.max(0, Math.round(totalSec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, "0")}`;
}

export function VoiceCreationModal({
  voices,
  activeVoiceId,
  training,
  ready,
  close,
  onSelectVoice,
  onRefresh,
}: {
  voices: StudioVoice[];
  activeVoiceId: string;
  training?: PvcTrainingState;
  ready: boolean;
  close: () => void;
  onSelectVoice: (voice: StudioVoice) => void;
  onRefresh: () => Promise<void>;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const previewAudio = useRef<HTMLAudioElement | null>(null);
  const [view, setView] = useState<"hub" | "instant" | "professional" | "design">("hub");
  const [libraryFilter, setLibraryFilter] = useState<"all" | "cloned" | "designed" | VoiceCategory>("all");
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [gender, setGender] = useState<"auto" | "male" | "female">("auto");
  const [pvcSteps, setPvcSteps] = useState(500);
  const [files, setFiles] = useState<UploadedItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [statusNote, setStatusNote] = useState("");

  // Voice Design state
  const [presetCat, setPresetCat] = useState<"all" | VoiceCategory>("all");
  const [designCategory, setDesignCategory] = useState<VoiceCategory>("mysterious");
  const [designPrompt, setDesignPrompt] = useState(DESIGN_PRESETS[0].prompt);
  const [designSample, setDesignSample] = useState(DESIGN_PRESETS[0].sample);
  const [designPitch, setDesignPitch] = useState<number>(DESIGN_PRESETS[0].pitch);
  const [designTexture, setDesignTexture] = useState<SculptTexture>(DESIGN_PRESETS[0].texture);
  const [designSeed, setDesignSeed] = useState(108);
  const [candidates, setCandidates] = useState<VoiceDesignCandidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState("");

  useEffect(() => {
    dialogRef.current?.showModal();
    return () => {
      previewAudio.current?.pause();
    };
  }, []);

  useEffect(() => {
    if (!training?.running) return;
    const id = setInterval(() => {
      void onRefresh();
    }, 2000);
    return () => clearInterval(id);
  }, [training?.running, onRefresh]);

  const totalSec = files.reduce((acc, item) => acc + item.seconds, 0);
  const totalMin = totalSec / 60;

  function playUrl(url: string, idKey: string) {
    if (!url) return;
    if (playingId === idKey) {
      previewAudio.current?.pause();
      setPlayingId(null);
      return;
    }
    previewAudio.current?.pause();
    const audio = new Audio(url);
    audio.onended = () => setPlayingId(null);
    audio.onerror = () => setPlayingId(null);
    previewAudio.current = audio;
    setPlayingId(idKey);
    void audio.play().catch(() => setPlayingId(null));
  }

  function togglePreview(voice: StudioVoice | undefined, idKey: string) {
    if (!voice?.preview_file) return;
    playUrl(`/media/voices/${voice.preview_file}`, idKey);
  }

  async function addFiles(list: FileList | null) {
    if (!list?.length) return;
    setError("");
    const next: UploadedItem[] = [];
    for (const f of Array.from(list)) {
      const sec = await probeAudioSeconds(f);
      next.push({ file: f, seconds: sec });
    }
    setFiles(prev => [...prev, ...next]);
  }

  function applyPreset(preset: DesignPreset) {
    setDesignPrompt(preset.prompt);
    setDesignSample(preset.sample);
    setDesignPitch(preset.pitch);
    setDesignTexture(preset.texture);
    setDesignCategory(preset.category);
    setGender(preset.gender);
    if (!name.trim() || DESIGN_PRESETS.some(p => p.label === name.trim())) {
      setName(preset.label);
    }
    setCandidates([]);
    setSelectedCandidateId("");
    setError("");
  }

  async function generateDesignPreviews() {
    if (!designPrompt.trim() || !designSample.trim()) return;
    setBusy(true);
    setError("");
    setStatusNote("Designing 3 unique character voice variations from your prompt…");
    previewAudio.current?.pause();
    setPlayingId(null);
    try {
      const res = await fetch("/api/studio-voices/design/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: designPrompt.trim(),
          sample_text: designSample.trim(),
          pitch_semitones: designPitch,
          texture: designTexture,
          seed: designSeed,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Voice design generation failed.");
      const list: VoiceDesignCandidate[] = Array.isArray(data.candidates) ? data.candidates : [];
      setCandidates(list);
      if (list[0]) {
        setSelectedCandidateId(list[0].id);
        playUrl(list[0].url, list[0].id);
      }
      setStatusNote("Audition the 3 generated character options below and save your favorite.");
    } catch (e) {
      setError(describe(e));
      setStatusNote("");
    } finally {
      setBusy(false);
    }
  }

  async function saveDesignedCharacter() {
    if (!name.trim() || !selectedCandidateId) return;
    setBusy(true);
    setError("");
    setStatusNote("Locking character timbre and saving to your Character Voices library…");
    try {
      const res = await fetch("/api/studio-voices/design/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          category: designCategory,
          prompt: designPrompt.trim(),
          candidate_id: selectedCandidateId,
          gender,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Failed to save character voice.");
      await onRefresh();
      if (data.voice) onSelectVoice(data.voice);
      setCandidates([]);
      setSelectedCandidateId("");
      setView("hub");
      setLibraryFilter("designed");
      setStatusNote(`Character Voice "${data.voice?.name}" saved and activated!`);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function submitInstant() {
    if (!name.trim() || !files.length) return;
    setBusy(true);
    setError("");
    setStatusNote("Analyzing pitch profile and creating Instant Voice Clone…");
    try {
      const fd = new FormData();
      fd.set("name", name.trim());
      fd.set("gender", gender);
      for (const item of files) fd.append("files", item.file);
      const res = await fetch("/api/studio-voices/instant", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Instant voice creation failed.");
      await onRefresh();
      if (data.voice) onSelectVoice(data.voice);
      setFiles([]);
      setName("");
      setView("hub");
      setStatusNote(`Instant Voice "${data.voice?.name}" created and activated!`);
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function submitProfessional() {
    if (!name.trim() || !files.length) return;
    if (totalSec < 1200) {
      setError(`Professional Voice Clone requires at least 20 minutes of audio (currently ${totalMin.toFixed(1)} min).`);
      return;
    }
    setBusy(true);
    setError("");
    setStatusNote("Uploading recordings and starting 2-stage Professional Voice Clone training…");
    try {
      const fd = new FormData();
      fd.set("name", name.trim());
      fd.set("steps", String(pvcSteps));
      for (const item of files) fd.append("files", item.file);
      const res = await fetch("/api/studio-voices/professional", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Failed to start PVC training.");
      await onRefresh();
      setFiles([]);
      setName("");
      setView("hub");
      setStatusNote("Professional Voice Clone training started in the background.");
    } catch (e) {
      setError(describe(e));
    } finally {
      setBusy(false);
    }
  }

  async function removeVoice(voiceId: string) {
    setError("");
    try {
      const res = await fetch(`/api/studio-voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Could not delete voice.");
      }
      await onRefresh();
    } catch (e) {
      setError(describe(e));
    }
  }

  const proPreviewVoice = voices.find(v => v.mode === "professional" && v.preview_file) ?? voices.find(v => v.preview_file);
  const instantPreviewVoice = voices.find(v => v.mode === "instant" && v.preview_file) ?? proPreviewVoice;
  const designPreviewVoice = voices.find(v => v.mode === "designed" && v.preview_file) ?? proPreviewVoice;

  const visiblePresets = useMemo(
    () => (presetCat === "all" ? DESIGN_PRESETS : DESIGN_PRESETS.filter(p => p.category === presetCat)),
    [presetCat]
  );

  const filteredVoices = useMemo(() => {
    return voices.filter(v => {
      if (libraryFilter === "all") return true;
      if (libraryFilter === "cloned") return v.mode === "instant" || v.mode === "professional";
      if (libraryFilter === "designed") return v.mode === "designed";
      return v.mode === "designed" && (v.category || "").toLowerCase() === libraryFilter;
    });
  }, [voices, libraryFilter]);

  const headerTitle =
    view === "hub"
      ? "Voice Creation"
      : view === "instant"
        ? "Instant Voice Clone"
        : view === "professional"
          ? "Professional Voice Clone"
          : "Voice Design";

  const headerSub =
    view === "hub"
      ? "Clone a real voice from audio or design brand-new human, mysterious, and cartoon character voices from a prompt."
      : view === "instant"
        ? "Upload 2–5 minutes of clean speech for rapid voice cloning with expressive pacing."
        : view === "professional"
          ? "Upload at least 20 minutes of clean audio to match exact expressions, modulations, articulations, and timbre."
          : "Generate a brand-new synthetic voice for a character or persona that never existed — no audio upload required.";

  return (
    <dialog
      ref={dialogRef}
      className="vc-modal"
      onCancel={e => {
        e.preventDefault();
        close();
      }}
    >
      <div className="vc-header">
        <div>
          <h2>{headerTitle}</h2>
          <p>{headerSub}</p>
        </div>
        <div className="vc-header-right">
          <span className="vc-count-pill">
            <i className="vc-count-dot" />
            {voices.length} {voices.length === 1 ? "voice" : "voices"}
          </span>
          {view !== "hub" && (
            <button
              type="button"
              className="btn-inline"
              disabled={busy}
              onClick={() => {
                previewAudio.current?.pause();
                setPlayingId(null);
                setView("hub");
                setError("");
                setStatusNote("");
                setFiles([]);
              }}
            >
              <IconArrowLeft size={14} />
              <span>Back</span>
            </button>
          )}
          <button type="button" className="icon-btn" onClick={close} aria-label="Close voice creation">
            <IconClose size={15} />
          </button>
        </div>
      </div>

      {training?.running && (
        <div className="pvc-banner" role="status">
          <div className="pvc-banner-top">
            <strong className="btn-inline">
              <IconWaveform size={15} />
              <span>Training Professional Voice · {training.stage}</span>
            </strong>
            <span>{training.percent}%</span>
          </div>
          <div className="pvc-bar"><div style={{ width: `${Math.max(3, training.percent)}%` }} /></div>
          <small>{training.message}</small>
        </div>
      )}
      {training?.error && !training.running && (
        <p className="error" role="alert">Last PVC training error: {training.error}</p>
      )}

      {view === "hub" && (
        <>
          <div className="vc-cards vc-cards-three">
            {/* Card 1: Instant Voice Clone */}
            <article className="vc-card vc-card-instant">
              <div className="vc-card-copy">
                <div className="vc-card-title">
                  <h3>Instant Voice Clone</h3>
                  <span className="vc-badge instant">2–5 MIN</span>
                </div>
                <p>
                  Clone your voice fast with 2 to 5 minutes of clean sample audio. Upload audio files and start
                  generating immediately.
                </p>
              </div>
              <div className="vc-orb-stage">
                <svg className="vc-arc" viewBox="0 0 220 130" aria-hidden="true">
                  <path
                    d="M 24 118 A 86 86 0 0 1 196 118"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="2 7"
                    strokeLinecap="round"
                  />
                </svg>
                <button
                  type="button"
                  className="vc-orb orb-instant"
                  title="Preview instant voice"
                  aria-label={playingId === "card-instant" ? "Pause instant voice preview" : "Play instant voice preview"}
                  onClick={() => togglePreview(instantPreviewVoice, "card-instant")}
                >
                  <span className="vc-play-btn">
                    {playingId === "card-instant" ? <IconPause size={14} /> : <IconPlay size={14} />}
                  </span>
                </button>
                <div className="vc-gauge-labels">
                  <div><span>Min</span><strong>2 min</strong></div>
                  <div><span>Best</span><strong>5 min</strong></div>
                </div>
              </div>
              <button
                type="button"
                className="accent vc-card-cta"
                disabled={!ready}
                onClick={() => { setError(""); setStatusNote(""); setFiles([]); setName(""); setView("instant"); }}
              >
                Create Instant Voice
              </button>
            </article>

            {/* Card 2: Professional Voice Clone */}
            <article className="vc-card vc-card-pro">
              <div className="vc-card-copy">
                <div className="vc-card-title">
                  <h3>Professional Voice Clone</h3>
                  <span className="vc-badge pro">20+ MIN · PVC</span>
                </div>
                <p>
                  The most realistic voice clone. Learns your exact talking expressions, modulations, and 44.1 kHz
                  timbre from 20+ minutes of samples.
                </p>
              </div>
              <div className="vc-orb-stage">
                <svg className="vc-arc" viewBox="0 0 220 130" aria-hidden="true">
                  <path
                    d="M 24 118 A 86 86 0 0 1 196 118"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeDasharray="2 7"
                    strokeLinecap="round"
                  />
                </svg>
                <button
                  type="button"
                  className="vc-orb orb-pro"
                  title="Preview professional voice"
                  aria-label={playingId === "card-pro" ? "Pause professional voice preview" : "Play professional voice preview"}
                  onClick={() => togglePreview(proPreviewVoice, "card-pro")}
                >
                  <span className="vc-play-btn">
                    {playingId === "card-pro" ? <IconPause size={14} /> : <IconPlay size={14} />}
                  </span>
                </button>
                <div className="vc-gauge-labels">
                  <div><span>Min</span><strong>20 min</strong></div>
                  <div><span>Best</span><strong>30+ min</strong></div>
                </div>
              </div>
              <button
                type="button"
                className="vc-card-cta pro-cta"
                disabled={!ready || training?.running}
                onClick={() => { setError(""); setStatusNote(""); setFiles([]); setName(""); setView("professional"); }}
              >
                {training?.running ? "Training in Progress…" : "Train Professional Voice"}
              </button>
            </article>

            {/* Card 3: Voice Design (Prompt-to-Character) */}
            <article className="vc-card vc-card-design">
              <div className="vc-card-copy">
                <div className="vc-card-title">
                  <h3>Voice Design</h3>
                  <span className="vc-badge design">PROMPT</span>
                </div>
                <p>
                  Generate a voice for a human, mysterious villain, or cartoon character from a prompt and tweak it as
                  you like.
                </p>
              </div>
              <div className="vc-orb-stage vc-design-stage">
                <div className="vc-pill-cloud" aria-hidden="true">
                  <div className="vc-pill-row">
                    <span>{PILL_CLOUD[0]}</span>
                    <span>{PILL_CLOUD[1]}</span>
                  </div>
                  <div className="vc-pill-row">
                    <span>{PILL_CLOUD[2]}</span>
                    <span>{PILL_CLOUD[3]}</span>
                  </div>
                  <div className="vc-pill-row">
                    <span>{PILL_CLOUD[4]}</span>
                    <span>{PILL_CLOUD[5]}</span>
                  </div>
                  <div className="vc-pill-row">
                    <span>{PILL_CLOUD[6]}</span>
                    <span>{PILL_CLOUD[7]}</span>
                  </div>
                </div>
                <button
                  type="button"
                  className="vc-orb orb-design"
                  title="Preview designed character voice"
                  aria-label={playingId === "card-design" ? "Pause character preview" : "Play character preview"}
                  onClick={() => {
                    if (designPreviewVoice) {
                      togglePreview(designPreviewVoice, "card-design");
                    } else {
                      setError("");
                      setStatusNote("");
                      setName(DESIGN_PRESETS[0].label);
                      setView("design");
                    }
                  }}
                >
                  <span className="vc-play-btn">
                    {playingId === "card-design" ? <IconPause size={14} /> : <IconPlay size={14} />}
                  </span>
                </button>
              </div>
              <button
                type="button"
                className="vc-card-cta design-cta btn-inline"
                disabled={!ready || training?.running}
                onClick={() => {
                  setError("");
                  setStatusNote("");
                  setCandidates([]);
                  setSelectedCandidateId("");
                  if (!name.trim()) setName(DESIGN_PRESETS[0].label);
                  setView("design");
                }}
              >
                <IconWand size={14} />
                <span>Create New Character</span>
              </button>
            </article>
          </div>

          <section className="vc-library">
            <div className="vc-library-head">
              <div>
                <h3>Your Studio &amp; Character Voices</h3>
                <span>Click Use Voice to activate any cloned or designed character voice in the composer</span>
              </div>
              <div className="vc-filter-tabs" role="tablist" aria-label="Filter voices by category">
                <button type="button" aria-pressed={libraryFilter === "all"} onClick={() => setLibraryFilter("all")}>All</button>
                <button type="button" aria-pressed={libraryFilter === "cloned"} onClick={() => setLibraryFilter("cloned")}>Cloned</button>
                <button type="button" aria-pressed={libraryFilter === "designed"} onClick={() => setLibraryFilter("designed")}>Characters</button>
                <button type="button" aria-pressed={libraryFilter === "mysterious"} onClick={() => setLibraryFilter("mysterious")}>Mysterious</button>
                <button type="button" aria-pressed={libraryFilter === "cartoon"} onClick={() => setLibraryFilter("cartoon")}>Cartoon</button>
                <button type="button" aria-pressed={libraryFilter === "male"} onClick={() => setLibraryFilter("male")}>Male</button>
                <button type="button" aria-pressed={libraryFilter === "female"} onClick={() => setLibraryFilter("female")}>Female</button>
              </div>
            </div>
            <div className="vc-voice-list">
              {!filteredVoices.length && (
                <p className="hint vc-empty">No voices in this category yet. Click Create New Character above to design one.</p>
              )}
              {filteredVoices.map(v => {
                const isActive = v.id === activeVoiceId;
                const orbClass =
                  v.mode === "professional"
                    ? "vc-mini-orb orb-pro"
                    : v.mode === "designed"
                      ? "vc-mini-orb orb-design"
                      : "vc-mini-orb orb-instant";
                const badgeClass =
                  v.mode === "professional"
                    ? "vc-badge pro"
                    : v.mode === "designed"
                      ? "vc-badge design"
                      : "vc-badge instant";
                const badgeLabel =
                  v.mode === "professional"
                    ? "PRO"
                    : v.mode === "designed"
                      ? "CHARACTER"
                      : "INSTANT";
                return (
                  <div key={v.id} className={isActive ? "vc-voice-row active" : "vc-voice-row"}>
                    <button
                      type="button"
                      className={orbClass}
                      disabled={!v.preview_file}
                      onClick={() => togglePreview(v, v.id)}
                      title={v.preview_file ? "Play preview" : "Built-in narrator"}
                      aria-label={playingId === v.id ? `Pause ${v.name} preview` : `Play ${v.name} preview`}
                    >
                      {playingId === v.id ? <IconPause size={12} /> : <IconPlay size={12} />}
                    </button>
                    <div className="vc-voice-meta">
                      <div className="vc-voice-name">
                        <strong>{v.name}</strong>
                        <span className={badgeClass}>{badgeLabel}</span>
                        {v.category ? <span className="vc-chip">{v.category}</span> : null}
                        {v.target_f0_hz ? <span className="vc-chip">{Math.round(v.target_f0_hz)} Hz</span> : null}
                        {v.mode !== "designed" && v.sample_minutes ? (
                          <span className="vc-chip">{v.sample_minutes.toFixed(1)} min</span>
                        ) : null}
                      </div>
                      <small>{v.description || "Ready for local synthesis"}</small>
                    </div>
                    <div className="vc-voice-actions">
                      <button
                        type="button"
                        className={isActive ? "accent btn-inline" : "btn-inline"}
                        onClick={() => { onSelectVoice(v); close(); }}
                      >
                        {isActive ? <><IconCheck size={13} /><span>Active</span></> : <span>Use Voice</span>}
                      </button>
                      {!v.is_default && (
                        <button
                          type="button"
                          className="icon-btn danger-hover"
                          title="Delete voice"
                          aria-label={`Delete ${v.name}`}
                          onClick={() => void removeVoice(v.id)}
                        >
                          <IconTrash size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        </>
      )}

      {view === "design" && (
        <div className="vc-form vc-design-form">
          <div className="vc-preset-section">
            <div className="vc-preset-header">
              <span className="vc-section-label">Character Archetypes &amp; Starter Prompts</span>
              <div className="vc-filter-tabs">
                <button type="button" aria-pressed={presetCat === "all"} onClick={() => setPresetCat("all")}>
                  All (16)
                </button>
                <button type="button" className="btn-inline" aria-pressed={presetCat === "mysterious"} onClick={() => setPresetCat("mysterious")}>
                  <IconEye size={13} /><span>Mysterious</span>
                </button>
                <button type="button" className="btn-inline" aria-pressed={presetCat === "cartoon"} onClick={() => setPresetCat("cartoon")}>
                  <IconSparkle size={13} /><span>Cartoon &amp; Creature</span>
                </button>
                <button type="button" className="btn-inline" aria-pressed={presetCat === "male"} onClick={() => setPresetCat("male")}>
                  <IconUser size={13} /><span>Male</span>
                </button>
                <button type="button" className="btn-inline" aria-pressed={presetCat === "female"} onClick={() => setPresetCat("female")}>
                  <IconUser size={13} /><span>Female</span>
                </button>
              </div>
            </div>
            <div className="vc-preset-grid">
              {visiblePresets.map(p => (
                <button
                  key={p.id}
                  type="button"
                  disabled={busy}
                  className={designPrompt === p.prompt ? "vc-preset-chip active" : "vc-preset-chip"}
                  onClick={() => applyPreset(p)}
                >
                  <strong>{p.label}</strong>
                  <span>{p.category}</span>
                </button>
              ))}
            </div>
          </div>

          <label>
            Voice description prompt (describe age, gender, pitch, rasp, accent, &amp; personality)
            <textarea
              rows={3}
              disabled={busy}
              value={designPrompt}
              onChange={e => setDesignPrompt(e.target.value)}
              placeholder="e.g. A deep, velvety mysterious villain speaking slowly from the shadows with dark baritone gravitas…"
            />
          </label>

          <label>
            Audition script (spoken to preview your 3 generated voice variations)
            <textarea
              rows={2}
              disabled={busy}
              value={designSample}
              onChange={e => setDesignSample(e.target.value)}
              placeholder="Enter a short line in character to audition the voice…"
            />
          </label>

          <div className="vc-form-grid vc-tweak-grid">
            <label>
              Character vocal sculpt
              <select
                value={designTexture}
                disabled={busy}
                onChange={e => setDesignTexture(e.target.value as SculptTexture)}
              >
                <option value="natural">Natural Human Vocal Tract</option>
                <option value="cartoon">Cartoon / Animated Brightness</option>
                <option value="gritty">Gritty / Raspy Edge</option>
                <option value="deep_creature">Colossal Bass / Creature</option>
                <option value="ethereal">Ethereal / Haunting Shimmer</option>
              </select>
            </label>
            <label>
              Formant &amp; pitch tweak{" "}
              <output>{designPitch > 0 ? `+${designPitch.toFixed(1)}` : designPitch.toFixed(1)} st</output>
              <input
                type="range"
                min={-5}
                max={5}
                step={0.2}
                disabled={busy}
                value={designPitch}
                onChange={e => setDesignPitch(e.target.valueAsNumber)}
              />
            </label>
          </div>

          <div className="vc-design-actions">
            <button
              type="button"
              className="btn-inline"
              disabled={busy}
              onClick={() => setDesignSeed(Math.floor(Math.random() * 100000))}
            >
              <IconShuffle size={14} />
              <span>Seed #{designSeed}</span>
            </button>
            <button
              type="button"
              className="accent btn-inline"
              disabled={busy || !ready || designPrompt.trim().length < 4 || designSample.trim().length < 6}
              onClick={() => void generateDesignPreviews()}
            >
              <IconWand size={14} />
              <span>{busy ? "Designing 3 Character Voices…" : "Generate 3 Voice Options"}</span>
            </button>
          </div>

          {candidates.length > 0 && (
            <div className="vc-candidates-panel">
              <div className="vc-candidates">
                {candidates.map(cand => {
                  const chosen = cand.id === selectedCandidateId;
                  const isPlaying = playingId === cand.id;
                  return (
                    <div
                      key={cand.id}
                      className={chosen ? "vc-cand-card selected" : "vc-cand-card"}
                      onClick={() => setSelectedCandidateId(cand.id)}
                    >
                      <button
                        type="button"
                        className="vc-mini-orb orb-design"
                        aria-label={isPlaying ? `Pause ${cand.label}` : `Play ${cand.label}`}
                        onClick={e => {
                          e.stopPropagation();
                          setSelectedCandidateId(cand.id);
                          playUrl(cand.url, cand.id);
                        }}
                      >
                        {isPlaying ? <IconPause size={12} /> : <IconPlay size={12} />}
                      </button>
                      <div className="vc-cand-meta">
                        <strong>{cand.label}</strong>
                        <small>{cand.seconds.toFixed(1)}s · Seed {cand.seed}</small>
                      </div>
                      {chosen && <IconCheck size={14} className="ok-text" />}
                    </div>
                  );
                })}
              </div>

              <div className="vc-save-character-bar">
                <label>
                  Character voice name
                  <input
                    value={name}
                    disabled={busy}
                    placeholder="e.g. Shadow Mastermind"
                    onChange={e => setName(e.target.value)}
                  />
                </label>
                <label>
                  Category
                  <select
                    value={designCategory}
                    disabled={busy}
                    onChange={e => setDesignCategory(e.target.value as VoiceCategory)}
                  >
                    <option value="mysterious">Mysterious &amp; Villain</option>
                    <option value="cartoon">Cartoon &amp; Creature</option>
                    <option value="male">Male Character</option>
                    <option value="female">Female Character</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="accent btn-inline vc-save-btn"
                  disabled={busy || !name.trim() || !selectedCandidateId}
                  onClick={() => void saveDesignedCharacter()}
                >
                  <IconCheck size={14} />
                  <span>Save Character Voice</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {view === "instant" && (
        <div className="vc-form">
          <div className="vc-form-grid">
            <label>
              Voice name
              <input
                value={name}
                disabled={busy}
                placeholder="e.g. Alex · Creator Voice"
                onChange={e => setName(e.target.value)}
              />
            </label>
            <label>
              Vocal character (Actor matching)
              <select value={gender} disabled={busy} onChange={e => setGender(e.target.value as "auto" | "male" | "female")}>
                <option value="auto">Auto-detect from audio pitch</option>
                <option value="male">Male expressive actor</option>
                <option value="female">Female expressive actor</option>
              </select>
            </label>
          </div>

          <label className="vc-dropzone">
            <IconUpload size={20} className="dropzone-icon" />
            <strong>Upload clean audio recordings (2–5 minutes recommended)</strong>
            <span>WAV, MP3, M4A, FLAC · Single or multiple files · Quiet room, no background music</span>
            <input
              type="file"
              multiple
              disabled={busy}
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={e => void addFiles(e.target.files)}
            />
          </label>

          <div className="vc-meter-card">
            <div className="vc-meter-row">
              <span>Uploaded audio duration</span>
              <strong className={totalSec >= 120 ? "ok-text btn-inline" : "btn-inline"}>
                <span>{formatMinSec(totalSec)} ({totalMin.toFixed(1)} min) ·</span>
                {totalSec >= 120 ? (
                  <><IconCheck size={13} /><span>Ideal 2–5 min range</span></>
                ) : totalSec >= 15 ? (
                  <span>Usable (2+ min recommended)</span>
                ) : (
                  <span>Min 15s required</span>
                )}
              </strong>
            </div>
            <div className="pvc-bar">
              <div style={{ width: `${Math.min(100, (totalSec / 300) * 100)}%` }} />
            </div>
          </div>

          {files.length > 0 && (
            <ul className="vc-file-list">
              {files.map((item, idx) => (
                <li key={`${item.file.name}-${idx}`}>
                  <span>{item.file.name}</span>
                  <span className="vc-file-right">
                    <span>{formatMinSec(item.seconds)}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      aria-label={`Remove ${item.file.name}`}
                      disabled={busy}
                      onClick={() => setFiles(prev => prev.filter((_, i) => i !== idx))}
                    >
                      <IconClose size={12} />
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="vc-form-footer">
            <button
              type="button"
              className="accent"
              disabled={busy || !name.trim() || totalSec < 15}
              onClick={() => void submitInstant()}
            >
              {busy ? "Creating Instant Voice…" : "Create Instant Voice"}
            </button>
          </div>
        </div>
      )}

      {view === "professional" && (
        <div className="vc-form">
          <div className="vc-form-grid">
            <label>
              Voice name
              <input
                value={name}
                disabled={busy}
                placeholder="e.g. Alex · Studio PVC"
                onChange={e => setName(e.target.value)}
              />
            </label>
            <label>
              Training depth (local Apple Silicon)
              <select value={pvcSteps} disabled={busy} onChange={e => setPvcSteps(Number(e.target.value))}>
                <option value={500}>Standard PVC · 500 steps (~25 min)</option>
                <option value={800}>High Likeness · 800 steps (~40 min)</option>
                <option value={1000}>Maximum PVC · 1000 steps (~50 min)</option>
              </select>
            </label>
          </div>

          <label className="vc-dropzone pro-dropzone">
            <IconUpload size={20} className="dropzone-icon" />
            <strong>Upload at least 20 minutes of your voice recordings</strong>
            <span>Include expressive talking, storytelling, and natural pauses · WAV / MP3 / M4A / FLAC (up to 1 GB total)</span>
            <input
              type="file"
              multiple
              disabled={busy}
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={e => void addFiles(e.target.files)}
            />
          </label>

          <div className="vc-meter-card">
            <div className="vc-meter-row">
              <span>Professional Voice Clone minimum</span>
              <strong className={totalSec >= 1200 ? "ok-text btn-inline" : "btn-inline"}>
                <span>{totalMin.toFixed(1)} / 20.0 min ·</span>
                {totalSec >= 1200 ? (
                  <><IconCheck size={13} /><span>Ready for 2-Stage PVC Training</span></>
                ) : (
                  <span>Need {(20 - totalMin).toFixed(1)} more min</span>
                )}
              </strong>
            </div>
            <div className="pvc-bar">
              <div style={{ width: `${Math.min(100, (totalSec / 1200) * 100)}%` }} />
            </div>
          </div>

          {files.length > 0 && (
            <ul className="vc-file-list">
              {files.map((item, idx) => (
                <li key={`${item.file.name}-${idx}`}>
                  <span>{item.file.name}</span>
                  <span className="vc-file-right">
                    <span>{formatMinSec(item.seconds)}</span>
                    <button
                      type="button"
                      className="icon-btn sm"
                      aria-label={`Remove ${item.file.name}`}
                      disabled={busy}
                      onClick={() => setFiles(prev => prev.filter((_, i) => i !== idx))}
                    >
                      <IconClose size={12} />
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}

          <div className="vc-form-footer">
            <button
              type="button"
              className="accent"
              disabled={busy || !name.trim() || totalSec < 1200 || training?.running}
              onClick={() => void submitProfessional()}
            >
              {busy ? "Uploading & Starting Training…" : "Start 2-Stage Professional Training"}
            </button>
          </div>
        </div>
      )}

      {statusNote && <p className="notice">{statusNote}</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </dialog>
  );
}
