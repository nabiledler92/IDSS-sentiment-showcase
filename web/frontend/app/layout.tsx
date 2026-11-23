import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import { Tabs } from "@/components/Tabs";
import info from "@/data/model-info.json";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const fraunces = Fraunces({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-fraunces", display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL("https://idss.nabil-takiyuddin.dev"),
  title: {
    default: "Kopi Sentiment IDSS | Nabil Takiyuddin",
    template: "%s | Kopi Sentiment IDSS",
  },
  description:
    "Classify customer review sentiment for kopi F&B SMEs and get an at-a-glance action recommendation. Logistic Regression with SMOTE, from Nabil Takiyuddin's Master's dissertation.",
  openGraph: { type: "website", siteName: "Kopi Sentiment IDSS", locale: "en_MY" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body>
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <div className="shell">
          <aside className="sidebar" aria-label="Active model">
            <h2>Active model</h2>
            <p className="model-pill">{info.deployedModel}</p>
            <p className="caption">Default model trained on Kopi Kenangan app reviews.</p>
            <p className="caption">Classes: Negative, Neutral, Positive</p>
            <hr />
            <div className="about">
              <p className="caption">
                Built from Nabil Takiyuddin&apos;s Master of Data Science dissertation.{" "}
                <a href="https://nabil-takiyuddin.dev/projects/kopi-sentiment-idss">Read the case study</a>
                {" · "}
                <a href="https://github.com/nabiledler92/IDSS-sentiment-showcase">Source code</a>
              </p>
              <p className="caption" style={{ marginTop: 12 }}>
                To train fresh models on your own review export, use the{" "}
                <a href="https://kopi-sentiment-idss.streamlit.app/">Streamlit version</a>&apos;s Train on Your Own Data tab.
              </p>
            </div>
          </aside>
          <main id="main" className="main">
            <h1>Knowledge-Driven IDSS for Malaysian Kopi F&amp;B SME Product Managers</h1>
            <p className="lede">Classify customer review sentiment and get an at-a-glance action recommendation.</p>
            <Tabs />
            <div className="tab-body">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
