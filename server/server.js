const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

 // It should print the actual key


const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("MongoDB connected"))
.catch((err) => console.log("MongoDB connection error:", err));

app.use('/api/questions', require('./routes/questions'));
// in backend/index.js or server.js
app.use("/api/agent", require("./routes/levelAgent"));


const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
