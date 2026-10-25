import { EventSourceParserStream } from "eventsource-parser/stream";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function parseData(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

// POST /ask/stream  { query, mode }  ->  SSE stream.
// Yields { event, data }. Expected events:
//   (default/"token")  text chunk: "text" or { text }
//   "sources"          [{ file, chapter, text, score }]
//   "refusal"          reason text (answer not supported by the documents)
//   "error"            message
//   "done"             end of stream
export async function* sendChatMessage({ query, mode, signal }) {
  let response;
  try {
    response = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode }),
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(`Cannot reach the document API at ${API_URL}. Check that the backend is running.`);
  }

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string" ? body.detail : `The document API returned an error (${response.status}).`
    );
  }

  const stream = response.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream());

  for await (const e of stream) {
    // eventsource-parser v2 adds `type`; v3 does not. Accept both.
    if (e.type && e.type !== "event") continue;
    yield { event: e.event || "message", data: parseData(e.data) };
  }
}
