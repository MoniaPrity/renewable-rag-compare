import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        vector: "#2563eb",
        graph: "#16a34a",
      },
    },
  },
  plugins: [],
};

export default config;
