const express = require("express");
const router = express.Router();
const { GoogleGenerativeAI } = require("@google/generative-ai");
require("dotenv").config(); // Make sure your .env is at root

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/decide-level", async (req, res) => {
  try {
    const { score, emotion, level } = req.body;

    if (!score || !emotion || !level) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    const prompt = `
You are an intelligent agent for an adaptive learning app.

Rules:
- Based on the current level (${level}), the user's quiz score (${score}/5), and their emotion (${emotion}), decide what level they should be in next.
- Available levels: level1 (easy), level2 (medium), level3 (hard)
-If score < 3 and emotion is angry, always go down one level.
-If score >= 4 and emotion is happy, always go up one level.
-If score is 3, stay on the same level unless emotion is angry.

- ONLY return one word: level1, level2, or level3. No explanation.

What level should the user go to next?
`;

    const model = genAI.getGenerativeModel({ model: "models/gemini-1.5-flash" });
    const result = await model.generateContent(prompt);
    const text = result.response.text().trim().toLowerCase();

    console.log("Gemini raw response:", text);

    const cleaned = text.replace(/[^a-z0-9]/gi, "");
    const validLevels = ["level1", "level2", "level3"];

    if (validLevels.includes(cleaned)) {
      return res.json({ nextLevel: cleaned });
    } else {
      console.error("Invalid Gemini response:", text);
      return res.status(400).json({ error: "Unexpected Gemini output", raw: text });
    }

  } catch (err) {
    console.error("Gemini Error:", err);
    return res.status(500).json({ error: "Gemini API call failed", details: err.message });
  }
});

module.exports = router;
