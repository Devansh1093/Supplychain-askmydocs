import { useEffect, useRef } from "react";

// Keeps the page scrolled to the bottom while an answer streams.
// Pauses if the user scrolls up; resumes when they return to the bottom
// or when a new answer starts. The whole document is the scroll container.
export default function useAutoScroll(isLoading, enabled) {
  const contentRef = useRef(null);
  const stick = useRef(true);

  useEffect(() => {
    if (isLoading) stick.current = true;
  }, [isLoading]);

  useEffect(() => {
    if (!enabled) return;
    const root = document.documentElement;
    const nearBottom = () => root.scrollHeight - root.scrollTop - root.clientHeight < 80;
    const onScroll = () => { stick.current = nearBottom(); };
    window.addEventListener("scroll", onScroll, { passive: true });

    const observer = new ResizeObserver(() => {
      if (stick.current) window.scrollTo({ top: root.scrollHeight });
    });
    if (contentRef.current) observer.observe(contentRef.current);

    return () => {
      window.removeEventListener("scroll", onScroll);
      observer.disconnect();
    };
  }, [enabled]);

  return contentRef;
}
