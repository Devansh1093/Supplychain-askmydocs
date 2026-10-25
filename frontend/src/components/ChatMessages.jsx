import { useState } from "react";
import Markdown from "react-markdown";
import Spinner from "./Spinner";
import Sources from "./Sources";

// Turn [1], [2] markers into links we can render as clickable citation badges.
const linkCitations = (text) => text.replace(/\[(\d{1,2})\](?!\()/g, "[$1](#cite-$1)");

function AssistantMessage({ msg }) {
  const [active, setActive] = useState(null);
  const count = msg.sources?.length ?? 0;

  const components = {
    a({ href, children }) {
      const m = /^#cite-(\d+)$/.exec(href || "");
      if (m) {
        const n = Number(m[1]);
        if (n < 1 || n > count) return <span>[{n}]</span>;
        return (
          <button
            type="button"
            onClick={() => setActive({ index: n - 1, tick: Date.now() })}
            aria-label={`Show source ${n}`}
            className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-mist px-1 align-text-top text-xs font-semibold text-teal hover:bg-teal hover:text-white"
          >
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" className="text-teal underline">
          {children}
        </a>
      );
    },
  };

  if (msg.loading && !msg.content) return <Spinner />;

  return (
    <div>
      {msg.declined ? (
        <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm">
          <p className="font-semibold">Not found in the documents</p>
          <p className="mt-1 text-ink/80">{msg.content}</p>
        </div>
      ) : (
        <div className="prose">
          <Markdown components={components}>{linkCitations(msg.content)}</Markdown>
        </div>
      )}
      {msg.error && (
        <p role="alert" className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-800">
          {msg.error}
        </p>
      )}
      <Sources sources={msg.sources} active={active} />
    </div>
  );
}

export default function ChatMessages({ messages }) {
  return (
    <div className="space-y-8">
      {messages.map((msg, i) =>
        msg.role === "user" ? (
          <div key={i} className="flex justify-end">
            <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-mist px-4 py-2.5 text-[0.95rem]">
              {msg.content}
            </p>
          </div>
        ) : (
          <AssistantMessage key={i} msg={msg} />
        ),
      )}
    </div>
  );
}
