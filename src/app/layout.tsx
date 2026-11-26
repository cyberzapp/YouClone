import type { Metadata } from "next";
import "./studio.css";

export const metadata: Metadata = {
  title: "YouClone",
  description: "Local Apple Silicon AI Voice Cloning & Expressive Speech Studio",
  icons: { icon: "/favicon.png" },
};

export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
