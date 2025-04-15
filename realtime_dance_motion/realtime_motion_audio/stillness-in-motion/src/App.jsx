import React, { useEffect, useState } from 'react';

function App() {
  const [motion, setMotion] = useState(0);
  const [volume, setVolume] = useState(0);
  const [score, setScore] = useState(0);
  const [label, setLabel] = useState("...");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMotion(data.motion);
      setVolume(data.volume);
      setScore(data.stillness_score);

      if (data.stillness_score > 0.7) setLabel("静寂");
      else if (data.stillness_score > 0.4) setLabel("中間");
      else setLabel("騒がしい");
    };

    return () => ws.close();
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>Stillness in Motion</h1>
      <p><strong>動き:</strong> {motion.toFixed(3)}</p>
      <p><strong>音量:</strong> {volume.toFixed(3)}</p>
      <p><strong>スコア:</strong> {score.toFixed(3)}</p>
      <p><strong>空気感:</strong> {label}</p>
    </div>
  );
}

export default App;