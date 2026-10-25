import { useEffect, useState } from "react";

function normalize(s) {
  const file = String(s.file ?? s.source ?? s.doc ?? s.document ?? "Unknown document");
  const score = s.rerank_score ?? s.score;
  return {
    // Citations show file + chapter only (no page numbers).
    file: file.split(/[\\/]/).pop().replace(/\.pdf$/i, ""),
    chapter: s.chapter ?? s.section ?? s.heading ?? "",
    text: s.text ?? s.chunk ?? s.content ?? "",
    score: typeof score === "number" ? score : null,
  };
}

// `active` is { index, tick }: set by clicking a [n] marker in the answer.
export default function Sources({ sources, active }) {
  const [open, setOpen] = useState(null);

  useEffect(() => {
    if (active) setOpen(active.index);
  }, [active]);

  if (!sources?.length) return null;

  return (
    <section className="mt-4" aria-label="Sources">
      <h3 className="mb-2 text-sm font-semibold text-ink/70">Sources</h3>
      <ol className="space-y-2">
        {sources.map((raw, i) => {
          const s = normalize(raw);
          const isOpen = open === i;
          return (
            <li key={i} className={`rounded-lg border ${isOpen ? "border-teal" : "border-ink/15"}`}>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : i)}
                aria-expanded={isOpen}
                className="flex w-full items-start gap-3 px-3 py-2 text-left"
              >
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-teal text-xs font-semibold text-white">
                  {i + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">{s.file}</span>
                  {s.chapter && <span className="block truncate text-xs text-ink/60">{s.chapter}</span>}
                </span>
                {s.score !== null && (
                  <span className="shrink-0 text-xs text-ink/50">relevance {s.score.toFixed(2)}</span>
                )}
              </button>
              {isOpen && s.text && (
                <p className="whitespace-pre-wrap border-t border-ink/10 bg-mist px-3 py-2 text-sm leading-relaxed">
                  {s.text}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
