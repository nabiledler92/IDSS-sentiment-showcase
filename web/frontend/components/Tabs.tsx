"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Batch Analyze" },
  { href: "/review", label: "Analyze a Review" },
  { href: "/model", label: "Model Info" },
];

export function Tabs() {
  const pathname = usePathname();
  return (
    <nav className="tabs" aria-label="Sections">
      {TABS.map((t) => (
        <Link key={t.href} href={t.href} aria-current={pathname === t.href ? "page" : undefined}>
          {t.label}
        </Link>
      ))}
    </nav>
  );
}
