const express = require('express');
const router = express.Router();
const Question = require('../models/Question');

// Get the available topics for a given level
router.get('/topics/:level', async (req, res) => {
  const { level } = req.params;
  try {
    const doc = await Question.findOne();
    if (doc && doc[level]) {
      // Convert the Map keys into an array
      const topics = Array.from(doc[level].keys());
      return res.json(topics);
    }
    res.status(404).send("No topics found for the specified level");
  } catch (err) {
    res.status(500).send(err.message);
  }
});

// Get questions for a given level and topic
router.get('/:level/:topic', async (req, res) => {
  const { level, topic } = req.params;
  try {
    const doc = await Question.findOne();
    if (doc && doc[level] && doc[level].get(topic)) {
      return res.json(doc[level].get(topic));
    }
    res.status(404).send("No questions found for the specified level and topic");
  } catch (err) {
    res.status(500).send("Server error: " + err.message);
  }
});

module.exports = router;
