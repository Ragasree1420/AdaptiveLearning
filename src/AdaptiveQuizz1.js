import React, { useState, useEffect } from "react";
import "./AdaptiveQuizz1.css";
import questionsData from "./questions.json";

function AdaptiveQuizz1() {
  const [level, setLevel] = useState("level2");
  const [topic, setTopic] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [score, setScore] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [levelMessage, setLevelMessage] = useState("");
  const [showGoButton, setShowGoButton] = useState(false);
  const[feedback,setFeedBack]=useState(null);

  useEffect(() => {
    if (topic) {
      setQuestions(shuffleArray(questionsData[level][topic]).slice(0, 5));
    }
  }, [level, topic]);

  function shuffleArray(array) {
    return array.sort(() => Math.random() - 0.5);
  }

  const handleAnswerClick = (option) => {
    setSelectedAnswer(option);
  };

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

  const resetTopicSelection = () => {
    setTopic(null);
    setCurrentQuestionIndex(0);
    setScore(0);
    setShowScore(false);
    setLevelMessage("");
    setShowGoButton(false);
  };
  const handleGoToNextLevel = () => {
    setShowScore(false); // Hide score screen
    setLevelMessage(""); // Clear level message
    setShowGoButton(false); // Hide Go button
    setCurrentQuestionIndex(0); // Reset quiz progress
    setScore(0); // Reset score
    setSelectedAnswer(null); // Reset answer selection
  
    // Increase level if possible
    let nextLevel = level;
  
  if (score >= 3 && feedback === "happy") {
    if (level === "level1") nextLevel = "level2";
    else if (level === "level2") nextLevel = "level3";
  } else if (score < 3 && feedback === "angry") {
    if (level === "level3") nextLevel = "level2";
    else if (level === "level2") nextLevel = "level1";
  }

  setLevel(nextLevel);
  
    // Load new set of random questions
    setQuestions(shuffleArray(questionsData[nextLevel][topic]).slice(0, 5));
  };
  

  return (
    <div className="quiz-background">
      <div className="quiz-container">
        <h1>Adaptive Learning Quiz</h1>

        {!topic ? (
          <div>
            <h3>Select a Topic:</h3>
            {Object.keys(questionsData[level]).map((subject) => (
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
