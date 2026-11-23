/*
 * Renders the small subset of markdown the model writes (paragraphs, "- " and
 * "1. " list items, **bold**) as React elements. No HTML is ever injected, so
 * model output can't add markup or scripts to the page.
 */
function inline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith("**") && p.endsWith("**") && p.length > 4 ? <strong key={i}>{p.slice(2, -2)}</strong> : <span key={i}>{p}</span>,
  );
}

export function Prose({ text }: { text: string }) {
  const blocks: React.ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {list.map((li, i) => (
            <li key={i}>{inline(li)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      list.push(bullet[1]);
      continue;
    }
    flush();
    blocks.push(<p key={`p-${blocks.length}`}>{inline(line.replace(/^#+\s*/, ""))}</p>);
  }
  flush();
  return <div className="prose">{blocks}</div>;
}
