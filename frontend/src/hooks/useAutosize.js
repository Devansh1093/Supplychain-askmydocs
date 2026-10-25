import { useEffect, useRef } from "react";

// Grows a textarea with its content, up to maxHeight px.
export default function useAutosize(value, maxHeight = 160) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [value, maxHeight]);
  return ref;
}
