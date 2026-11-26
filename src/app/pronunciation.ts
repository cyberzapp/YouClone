export function applyPronunciation(text: string, rules = "") {
  const aliases = new Map<string, string>();
  for (const line of rules.split("\n")) {
    const equals = line.indexOf("=");
    if (equals < 0) continue;
    const term = line.slice(0, equals).trim();
    const spoken = line.slice(equals + 1).trim();
    if (term && spoken) aliases.set(term, spoken);
  }
  if (!aliases.size) return text.replace(/\[\[([^\[\]\n]+)\]\]/g, "$1");
  const terms = [...aliases.keys()].sort((a, b) => b.length - a.length)
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const pattern = new RegExp(`\\[\\[[^\\[\\]\\n]+\\]\\]|\\[[^\\]\\n]*\\]|<break\\b[^>]*>|(?<![\\p{L}\\p{M}\\p{N}_])(?:${terms})(?![\\p{L}\\p{M}\\p{N}_])`, "gu");
  return text.replace(pattern, term => {
    if (term.startsWith("[[") && term.endsWith("]]")) return term.slice(2, -2);
    return aliases.get(term) ?? term;
  });
}
