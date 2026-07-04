import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Orbit Wars — An AI Competition Journal",
  description:
    "7 days. 23 scientists. 19 experiments. 491 leaderboard snapshots. Building an AI to compete in a Kaggle orbital-mechanics strategy game.",
  openGraph: {
    title: "Orbit Wars — An AI Competition Journal",
    description:
      "7 days. 23 scientists. 19 experiments. 491 leaderboard snapshots.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
    >
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
