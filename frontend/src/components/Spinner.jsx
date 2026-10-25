export default function Spinner() {
  return (
    <div className="flex items-center gap-1.5 py-2" role="status" aria-label="Searching the documents">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-teal motion-safe:animate-pulse"
          style={{ animationDelay: `${i * 180}ms` }}
        />
      ))}
    </div>
  );
}
