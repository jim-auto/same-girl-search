import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "same-girl-search",
  description: "Local face similarity search for lawful duplicate profile review"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}

