import type { Metadata } from "next";
import { ReviewForm } from "./ReviewForm";

export const metadata: Metadata = {
  title: "Analyze a review",
  description: "Classify a single customer review as Negative, Neutral or Positive, with a recommended action.",
};

export default function ReviewPage() {
  return (
    <>
      <h2>Analyze a single review</h2>
      <ReviewForm />
    </>
  );
}
