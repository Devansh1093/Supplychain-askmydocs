import useAutosize from "../hooks/useAutosize";

export default function ChatInput({ value, onChange, onSubmit, onStop, isLoading }) {
  const ref = useAutosize(value);

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (!isLoading) onSubmit();
    }
  }

  return (
    <div>
      <div className="flex items-end gap-2 rounded-2xl border border-ink/20 bg-white p-2 focus-within:border-teal">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your supply chain documents..."
          aria-label="Your question"
          className="max-h-40 flex-1 resize-none bg-transparent px-3 py-2 text-[0.95rem] outline-none placeholder:text-ink/40"
        />
        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl border border-ink/30 px-4 py-2 text-sm font-medium hover:bg-mist"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => onSubmit()}
            disabled={!value.trim()}
            className="rounded-xl bg-teal px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
      <p className="mt-2 px-1 text-xs text-ink/50">Enter to send. Shift + Enter for a new line.</p>
    </div>
  );
}
