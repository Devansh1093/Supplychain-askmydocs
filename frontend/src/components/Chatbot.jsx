import { useRef, useState } from "react";
import { useImmer } from "use-immer";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";
import useAutoScroll from "../hooks/useAutoScroll";
import { sendChatMessage } from "../api";

// `value` is sent to the backend as `mode`. Adjust to match your retriever.
const MODES = [
  { value: "hybrid_rerank", label: "Hybrid + rerank" },
  { value: "hybrid", label: "Hybrid (BM25 + vector)" },
  { value: "vector", label: "Vector only" },
  { value: "bm25", label: "Keyword only (BM25)" },
];

const EXAMPLES = [
  "How is average monthly consumption calculated?",
  "What is the purpose of buffer stock?",
  "How does a max-min inventory system work?",
];

const NO_ANSWER = "The retrieved passages do not support an answer to this question.";

const textOf = (d) => (typeof d === "string" ? d : (d?.text ?? d?.token ?? d?.content ?? ""));

export default function Chatbot() {
  const [messages, setMessages] = useImmer([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState(MODES[0].value);
  const abortRef = useRef(null);

  const isLoading = messages.length > 0 && !!messages[messages.length - 1].loading;
  const contentRef = useAutoScroll(isLoading, messages.length > 0);

  async function submit(text = input) {
    const query = text.trim();
    if (!query || isLoading) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setInput("");
    setMessages((d) => {
      d.push({ role: "user", content: query });
      d.push({ role: "assistant", content: "", sources: [], loading: true });
    });

    try {
      for await (const { event, data } of sendChatMessage({ query, mode, signal: controller.signal })) {
        if (event === "sources") {
          const list = Array.isArray(data) ? data : (data?.sources ?? []);
          setMessages((d) => { d[d.length - 1].sources = list; });
        } else if (event === "refusal" || event === "declined") {
          const t = textOf(data);
          setMessages((d) => {
            const m = d[d.length - 1];
            m.declined = true;
            if (t) m.content = t;
          });
        } else if (event === "error") {
          throw new Error(textOf(data) || "The backend reported an error.");
        } else if (event === "done") {
          break;
        } else {
          const t = textOf(data);
          if (t) setMessages((d) => { d[d.length - 1].content += t; });
        }
      }
      setMessages((d) => {
        const m = d[d.length - 1];
        m.loading = false;
        if (!m.content) {
          m.declined = true;
          m.content = NO_ANSWER;
        }
      });
    } catch (err) {
      setMessages((d) => {
        const m = d[d.length - 1];
        m.loading = false;
        if (err.name === "AbortError") {
          if (!m.content) m.content = "Stopped before an answer was generated.";
        } else {
          m.error = err.message;
        }
      });
    } finally {
      abortRef.current = null;
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-white text-ink">
      <header className="sticky top-0 z-10 border-b border-ink/10 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-4">
          <div>
            <h1 className="text-xl font-bold">SupplyDocsQA</h1>
            <p className="text-sm text-ink/60">Evidence-based supply chain answers</p>
          </div>
          <label className="text-right text-xs font-medium text-ink/60">
            Retrieval mode
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={isLoading}
              className="mt-1 block rounded-lg border border-ink/20 bg-white px-3 py-2 text-sm font-medium text-ink"
            >
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        {messages.length === 0 ? (
          <div className="pt-8">
            <h2 className="text-lg font-semibold">Ask a question about your documents</h2>
            <p className="mt-1 max-w-xl text-sm text-ink/70">
              Every answer cites the file and chapter it came from. If the documents don't support an answer, you'll
              be told instead of getting a guess.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              {EXAMPLES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => submit(q)}
                  className="rounded-full border border-teal/40 px-4 py-2 text-left text-sm text-teal hover:bg-mist"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div ref={contentRef}>
            <ChatMessages messages={messages} />
          </div>
        )}
      </main>

      <footer className="sticky bottom-0 border-t border-ink/10 bg-white/95 backdrop-blur">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={submit}
            onStop={() => abortRef.current?.abort()}
            isLoading={isLoading}
          />
        </div>
      </footer>
    </div>
  );
}
