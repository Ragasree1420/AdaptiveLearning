import React, { useState, useEffect } from "react";
import "./AdaptiveQuizz1.css";
import axios from 'axios';

function AdaptiveQuizz1() {
  const [level, setLevel] = useState("level2");
  const [topic, setTopic] = useState(null);
  const [topics, setTopics] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [score, setScore] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [levelMessage, setLevelMessage] = useState("");
  const [showGoButton, setShowGoButton] = useState(false);
  const [feedback, setFeedBack] = useState(null);

  // Fetch topics based on level
  useEffect(() => {
    axios
      .get(`http://localhost:5000/api/questions/topics/${level}`)
      .then((res) => setTopics(res.data))
      .catch((err) => console.error("Error fetching topics", err));
  }, [level]);

  // Fetch questions when topic or level changes
  useEffect(() => {
    if (topic) {
      axios
        .get(`http://localhost:5000/api/questions/${level}/${topic}`)
        .then((res) => {
          // Shuffle questions and take the first 5
          setQuestions(shuffleArray(res.data).slice(0, 5));
        })
        .catch((err) => console.error("Error fetching questions:", err));
    }
  }, [level, topic]);

  function shuffleArray(array) {
    return array.sort(() => Math.random() - 0.5);
  }

  // Handle answer selection
  const handleAnswerClick = (option) => {
    setSelectedAnswer(option);
  };

  // Handle next question after answer selection
  const handleNext = () => {
    if (selectedAnswer === questions[currentQuestionIndex].answer) {
      setScore(score + 1);
    }
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setSelectedAnswer(null);
    } else {
      setShowScore(true);
    }
  };

  // Handle feedback after quiz completion
  const handleFeedback = (feedback) => {
    setFeedBack(feedback);
    if (score >= 3 && feedback === "happy" && level !== "level3") {
      setLevel("level3");
      setLevelMessage("Congratulations! Moving to Level 3");
    } else if (score < 3 && (feedback === "neutral" || feedback === "happy")) {
      setLevelMessage("Stay on your current level. Keep improving!");
    } else if (score < 3 && feedback === "angry" && level !== "level1") {
      setLevel("level1");
      setLevelMessage("Moving down to Level 1. Try again!");
    }
    setShowGoButton(true);
  };

  // Reset topic and quiz state
  const resetTopicSelection = () => {
    setTopic(null);
    setCurrentQuestionIndex(0);
    setScore(0);
    setShowScore(false);
    setLevelMessage("");
    setShowGoButton(false);
  };

  // Handle the transition to the next level after feedback
  const handleGoToNextLevel = () => {
    setShowScore(false);
    setLevelMessage("");
    setShowGoButton(false);
    setCurrentQuestionIndex(0);
    setScore(0);
    setSelectedAnswer(null);

    let nextLevel = level;

    if (score >= 3 && feedback === "happy") {
      if (level === "level1") nextLevel = "level2";
      else if (level === "level2") nextLevel = "level3";
    } else if (score < 3 && feedback === "angry") {
      if (level === "level3") nextLevel = "level2";
      else if (level === "level2") nextLevel = "level1";
    }

    setLevel(nextLevel);
    
    // Fetch questions for the next level
    axios
      .get(`http://localhost:5000/api/questions/${nextLevel}/${topic}`)
      .then((res) => {
        setQuestions(shuffleArray(res.data).slice(0, 5));
      })
      .catch((err) => console.error("Error fetching next level questions:", err));
  };

  return (
    <div className="quiz-background">
      <div className="quiz-container">
        <h1>Adaptive Learning Quiz</h1>

        {/* Topic Selection */}
        {!topic ? (
          <div>
            <h3>Select a Topic:</h3>
            {topics.map((subject) => (
              <button
                key={subject}
                className="topic-button"
                onClick={() => setTopic(subject)}
              >
                {subject}
              </button>
            ))}
          </div>
        ) : showScore ? (
          // Show Score Screen with Feedback Options
          <div className="quiz-box">
            <h3>Your Score: {score}/{questions.length}</h3>
            <div className="feedback-buttons">
              <button onClick={() => handleFeedback("happy")} className="happy">😀 Happy</button>
              <button onClick={() => handleFeedback("neutral")} className="neutral">😐 Neutral</button>
              <button onClick={() => handleFeedback("angry")} className="angry">😠 Angry</button>
            </div>
            {levelMessage && <p className="level-message">{levelMessage}</p>}
            {showGoButton && <button className="go-button" onClick={handleGoToNextLevel}>Go to Next Level</button>}
          </div>
        ) : (
          // Show Question Box
          <div className="quiz-box">
            <button className="close-button" onClick={resetTopicSelection}>❌</button>
            <h3>{questions[currentQuestionIndex]?.question}</h3>
            {questions[currentQuestionIndex]?.options.map((option) => (
              <button
                key={option}
                className={`option-button ${selectedAnswer === option ? "selected" : ""}`}
                onClick={() => handleAnswerClick(option)}
              >
                {option}
              </button>
            ))}
            <button className="next-button" onClick={handleNext} disabled={!selectedAnswer}>
              Next
            </button>
            <div className="progress-bar" style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}></div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AdaptiveQuizz1;
