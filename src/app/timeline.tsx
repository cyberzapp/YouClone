"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import Regions, { type Region } from "wavesurfer.js/dist/plugins/regions.esm.js";
import type { Format } from "./types";

export type RegionRange = { start: number; end: number };
const clock = (seconds: number) =>
  `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;

export function Timeline({ audio, label, locked, canExportTake, canExportMaster, setPlayer, exportAudio }: {
  audio: string; label: string; locked: boolean; canExportTake: boolean; canExportMaster: boolean; setPlayer: (player: (() => void) | null) => void;
  exportAudio: (format: Format, subtitles: boolean, master: boolean, region?: RegionRange) => void;
}) {
  const surface = useRef<HTMLDivElement>(null);
  const wave = useRef<WaveSurfer | null>(null);
  const plugin = useRef<Regions | null>(null);
  const chosen = useRef<Region | null>(null);
  const looping = useRef(false);
  const [region, setRegion] = useState<RegionRange>();
  const [loop, setLoop] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [pitchLock, setPitchLock] = useState(true);
  const [format, setFormat] = useState<Format>("wav");
  const [subtitles, setSubtitles] = useState(false);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    if (!surface.current || !audio) return;
    const player = WaveSurfer.create({
      container: surface.current, height: 78, waveColor: "#777cbf", progressColor: "#46e4aa",
      cursorColor: "#ecfff8", barWidth: 2, barGap: 2, barRadius: 2, normalize: true,
    });
    const regions = player.registerPlugin(Regions.create());
    wave.current = player;
    plugin.current = regions;
    setPlayer(() => { void player.playPause().catch(e => setProblem(e.message)); });
    player.on("ready", value => setDuration(value));
    player.on("timeupdate", setTime);
    player.on("play", () => setPlaying(true));
    player.on("pause", () => setPlaying(false));
    player.on("finish", () => setPlaying(false));
    regions.on("region-created", selected => {
      if (chosen.current !== selected) chosen.current?.remove();
      chosen.current = selected;
      setRegion({ start: selected.start, end: selected.end });
    });
    regions.on("region-updated", selected => setRegion({ start: selected.start, end: selected.end }));
    regions.on("region-out", selected => {
      if (looping.current && chosen.current === selected) selected.play();
    });
    const unsubscribe = regions.enableDragSelection({ color: "#91a0fa42", minLength: .05 });
    void player.load(audio).catch(e => { if (e.name !== "AbortError") setProblem(e.message); });
    return () => {
      unsubscribe();
      player.destroy();
      wave.current = null;
      plugin.current = null;
      chosen.current = null;
      setPlayer(null);
    };
  }, [audio, setPlayer]);

  useEffect(() => { wave.current?.setPlaybackRate(rate, pitchLock); }, [rate, pitchLock]);

  return <section className="dock" aria-label="Waveform">
    <div className="dock-heading"><div><h2>Waveform</h2><span>{label || "Generate a take or open a reference"}</span></div>
      <output className="time">{clock(time)} <span>/ {clock(duration)}</span></output></div>
    <div ref={surface} className="wave">{!audio && <span>Audio will appear here</span>}</div>
    {problem && <p role="alert">{problem}</p>}
    <div className="transport">
      <button disabled={!duration} onClick={() => void wave.current?.playPause()}>{playing ? "Pause" : "Play"}</button>
      <button disabled={!duration} onClick={() => {
        const start = Math.min(wave.current?.getCurrentTime() ?? 0, Math.max(0, duration - .25));
        const selection = plugin.current?.addRegion({ start, end: Math.min(duration, start + 3), color: "#91a0fa42" });
        if (selection) setRegion({ start: selection.start, end: selection.end });
      }}>Set region</button>
      <button disabled={!region} aria-pressed={loop} onClick={() => {
        looping.current = !loop; setLoop(!loop);
        if (!loop) chosen.current?.play();
      }}>A/B loop</button>
      <button disabled={!region} onClick={() => { chosen.current?.remove(); chosen.current = null; setRegion(undefined); setLoop(false); looping.current = false; }}>Clear</button>
      <label>Playback <select value={rate} onChange={event => setRate(Number(event.target.value))}>
        {[.7, .85, 1, 1.15, 1.3].map(value => <option key={value} value={value}>{value}×</option>)}
      </select></label>
      <label><input type="checkbox" checked={pitchLock} onChange={event => setPitchLock(event.target.checked)} /> Preserve pitch</label>
      <div className="export-controls">
        <select aria-label="Export format" value={format} onChange={event => setFormat(event.target.value as Format)}>
          <option value="wav">WAV</option><option value="flac">FLAC</option><option value="mp3">MP3</option>
        </select>
        <label><input type="checkbox" checked={subtitles} onChange={event => setSubtitles(event.target.checked)} /> SRT</label>
        <button disabled={locked || !canExportTake} onClick={() => exportAudio(format, subtitles, false, region)}>{region ? "Export region" : label === "Master preview" ? "Export preview" : "Export take"}</button>
        <button className="accent" disabled={locked || !canExportMaster} onClick={() => exportAudio(format, subtitles, true)}>Export master</button>
      </div>
    </div>
  </section>;
}
