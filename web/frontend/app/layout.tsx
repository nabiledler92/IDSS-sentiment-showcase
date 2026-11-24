import type { Metadata } from "next";
import { GoogleTagManager } from "@next/third-parties/google";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Fraunces, Inter } from "next/font/google";
import { Tabs } from "@/components/Tabs";
import { TrackedLink } from "@/components/TrackedLink";
import info from "@/data/model-info.json";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const fraunces = Fraunces({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-fraunces", display: "swap" });

// Google Tag Manager container shared with nabil-takiyuddin.dev. Set
// NEXT_PUBLIC_GTM_ID to override; otherwise it loads on production builds
// only, so previews and local development don't send data.
const GTM_ID =
  process.env.NEXT_PUBLIC_GTM_ID || (process.env.VERCEL_ENV === "production" ? "GTM-MKG27HR8" : undefined);

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
      {GTM_ID ? <GoogleTagManager gtmId={GTM_ID} /> : null}
      <body>
        {GTM_ID ? (
          <noscript>
            <iframe
              src={`https://www.googletagmanager.com/ns.html?id=${GTM_ID}`}
              height="0"
              width="0"
              style={{ display: "none", visibility: "hidden" }}
              title="Google Tag Manager"
            />
          </noscript>
        ) : null}
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
                <TrackedLink href="https://nabil-takiyuddin.dev/projects/kopi-sentiment-idss" linkType="case_study">
                  Read the case study
                </TrackedLink>
                {" · "}
                <TrackedLink href="https://github.com/nabiledler92/IDSS-sentiment-showcase" linkType="source_code">
                  Source code
                </TrackedLink>
              </p>
              <p className="caption" style={{ marginTop: 12 }}>
                To train fresh models on your own review export, use the{" "}
                <TrackedLink href="https://kopi-sentiment-idss.streamlit.app/" linkType="streamlit">
                  Streamlit version
                </TrackedLink>
                &apos;s Train on Your Own Data tab.
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
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
