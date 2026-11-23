export type Label = "Negative" | "Neutral" | "Positive";
export const LABELS: Label[] = ["Negative", "Neutral", "Positive"];

export type Meta = {
  modelName: string;
  classes: string[];
  aiEnabled: boolean;
  aiModel: string;
  limits: { maxRows: number; maxFileMB: number };
};

export type Prediction = {
  label: Label;
  probabilities: Record<Label, number>;
  confidence: number;
  recommendation: string;
};

export type Columns = { columns: string[]; guess: string; rows: number; truncated: boolean };

export type Term = { term: string; count: number; pct_of_class: number; examples: string[] };

export type Stats = {
  total: number;
  counts: Record<Label, number>;
  negTerms: { term: string; count: number }[];
  posTerms: { term: string; count: number }[];
  negCategories: Record<string, number>;
};

export type BatchResult = {
  total: number;
  truncated: boolean;
  maxRows: number;
  counts: Record<Label, number>;
  negTerms: Term[];
  posTerms: Term[];
  categoriesAll: Record<string, number>;
  categoriesNegative: Record<string, number>;
  negatives: { text: string; confidence: number }[];
  negativeCount: number;
  stats: Stats;
  signature: string;
};

export type AIAnswer = { text: string; model: string };

export class ApiError extends Error {}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let message = `Something went wrong (HTTP ${res.status}). Try again.`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") message = body.detail;
    else if (Array.isArray(body?.detail)) message = "That input isn't valid. Check it and try again.";
  } catch {
    /* keep the generic message */
  }
  throw new ApiError(message);
}

export async function getMeta(): Promise<Meta> {
  return handle(await fetch("/api/meta"));
}

export async function predict(text: string, stopwords: boolean): Promise<Prediction> {
  return handle(
    await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, stopwords }),
    }),
  );
}

function fileForm(file: File, extra: Record<string, string> = {}) {
  const form = new FormData();
  form.append("file", file);
  for (const [k, v] of Object.entries(extra)) form.append(k, v);
  return form;
}

export async function readColumns(file: File): Promise<Columns> {
  return handle(await fetch("/api/columns", { method: "POST", body: fileForm(file) }));
}

export async function runBatch(file: File, textCol: string, stopwords: boolean): Promise<BatchResult> {
  return handle(
    await fetch("/api/batch", {
      method: "POST",
      body: fileForm(file, { text_col: textCol, stopwords: String(stopwords) }),
    }),
  );
}

export async function downloadBatchCsv(file: File, textCol: string, stopwords: boolean): Promise<Blob> {
  const res = await fetch("/api/batch/csv", {
    method: "POST",
    body: fileForm(file, { text_col: textCol, stopwords: String(stopwords) }),
  });
  if (!res.ok) await handle(res);
  return res.blob();
}

export async function getBrief(stats: Stats, signature: string): Promise<AIAnswer> {
  return handle(
    await fetch("/api/brief", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stats, signature }),
    }),
  );
}

export async function ask(stats: Stats, signature: string, question: string): Promise<AIAnswer> {
  return handle(
    await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stats, signature, question }),
    }),
  );
}

export const PALETTE: Record<Label, { accent: string; bg: string; text: string }> = {
  Negative: { accent: "#B4432F", bg: "#F6E3DF", text: "#6B2417" },
  Neutral: { accent: "#C99A3C", bg: "#F7EDD8", text: "#6B4F16" },
  Positive: { accent: "#4F7C4A", bg: "#E4EDE3", text: "#2C4A29" },
};
