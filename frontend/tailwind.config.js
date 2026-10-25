/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f2f3f",
        teal: "#087f7b",
        mist: "#edf6f6",
      },
    },
  },
  plugins: [],
};
