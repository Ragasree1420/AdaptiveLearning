const mongoose = require('mongoose');

// Define the schema for a question item
const questionItemSchema = new mongoose.Schema({
  question: { type: String, required: true },
  options: { type: [String], required: true },
  answer: { type: String, required: true }
});

// This main schema organizes the questions by level and by topic.
const questionSchema = new mongoose.Schema({
  level1: {
    // Using Map allows for flexible topics keys
    type: Map,
    of: [questionItemSchema],
    default: {}
  },
  level2: {
    type: Map,
    of: [questionItemSchema],
    default: {}
  },
  level3: {
    type: Map,
    of: [questionItemSchema],
    default: {}
  }
});

module.exports = mongoose.model('Question', questionSchema);
