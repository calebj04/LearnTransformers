import React, { useState, useEffect } from "react";

// Example vocabulary
const vocab = [
  "the",
  "a",
  "cat",
  "dog",
  "sat",
  "on",
  "mat",
  "runs",
  "fast",
  "slow",
];

// Utility to generate random hidden vector
const generateHiddenVector = (dim) =>
  Array.from({ length: dim }, () => Math.random() * 2 - 1);

// Linear projection: hidden -> logits
const linearProjection = (hidden, outputDim) => {
  const inputDim = hidden.length;
  const W = Array.from({ length: inputDim }, () =>
    Array.from({ length: outputDim }, () => Math.random() * 2 - 1)
  );
  const b = Array.from({ length: outputDim }, () => Math.random() * 2 - 1);

  const logits = Array.from(
    { length: outputDim },
    (_, j) => hidden.reduce((sum, h, i) => sum + h * W[i][j], 0) + b[j]
  );

  return logits;
};

// Softmax
const softmax = (logits, temperature = 1.0) => {
  const adjusted = logits.map((x) => x / temperature);
  const max = Math.max(...adjusted);
  const exps = adjusted.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
};

// Sampling strategies
const sampleNextToken = (probs, strategy, topK = 3) => {
  if (strategy === "greedy") {
    const maxIdx = probs.indexOf(Math.max(...probs));
    return maxIdx;
  } else if (strategy === "top-k") {
    const sorted = probs
      .map((p, idx) => ({ p, idx }))
      .sort((a, b) => b.p - a.p)
      .slice(0, topK);
    const total = sorted.reduce((sum, s) => sum + s.p, 0);
    const rnd = Math.random() * total;
    let acc = 0;
    for (let s of sorted) {
      acc += s.p;
      if (rnd <= acc) return s.idx;
    }
    return sorted[0].idx;
  } else {
    // default: stochastic sampling with temperature
    const rnd = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (rnd <= acc) return i;
    }
    return probs.indexOf(Math.max(...probs));
  }
};

const OutputProjectionModule = ({ inputTokens }) => {
  const hiddenDim = 16;
  const [hidden, setHidden] = useState(generateHiddenVector(hiddenDim));
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(3);
  const strategy = "top-k";
  const [logits, setLogits] = useState([]);
  const [probs, setProbs] = useState([]);
  const [selectedToken, setSelectedToken] = useState(null);

  // If parent passed tokens, use their words as vocabulary; otherwise use default vocab
  const vocabList =
    Array.isArray(inputTokens) && inputTokens.length
      ? inputTokens.map((t) =>
          typeof t === "string" ? t : t.word || String(t)
        )
      : vocab;

  // Project hidden -> logits -> probabilities
  useEffect(() => {
    const logitVals = linearProjection(hidden, vocabList.length);
    setLogits(logitVals);
    const probVals = softmax(logitVals, temperature);
    setProbs(probVals);
    const nextTokenIdx = sampleNextToken(probVals, "top-k", topK);
    setSelectedToken(nextTokenIdx);
  }, [hidden, temperature, topK]);

  return (
    <div style={{ padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h2>Output Projection Module Visualization</h2>
      <p>
        Hidden representations are projected to logits and transformed to a
        probability distribution. Adjust sampling strategies to see how the
        predicted next token changes.
      </p>

      <div style={{ marginBottom: 12 }}>
        <label>
          Top-K:{" "}
          <input
            type="number"
            value={topK}
            min={1}
            max={vocabList.length}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
      </div>

      <div style={{ marginBottom: 12 }}>
        <label>
          Temperature:{" "}
          <input
            type="range"
            min={0.1}
            max={3.0}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />{" "}
          {temperature.toFixed(1)}
        </label>
      </div>

      <h3>Probability Distribution</h3>
      <div
        style={{
          display: "flex",
          gap: 6,
          alignItems: "flex-end",
          height: 120,
          marginTop: 12,
        }}
      >
        {probs.map((p, i) => (
          <div
            key={i}
            title={`${vocabList[i]}: ${(p * 100).toFixed(2)}%`}
            style={{
              width: 24,
              height: `${p * 100}%`,
              backgroundColor: selectedToken === i ? "#f56a00" : "#3399ff",
              display: "flex",
              justifyContent: "center",
              alignItems: "flex-end",
              color: "var(--text)",
              fontSize: 10,
              borderRadius: 2,
            }}
          >
            <span style={{ transform: "rotate(-45deg)" }}>{vocabList[i]}</span>
          </div>
        ))}
      </div>

      {selectedToken !== null && (
        <div style={{ marginTop: 12, fontSize: 16 }}>
          Predicted next token: <strong>{vocabList[selectedToken]}</strong>
        </div>
      )}
    </div>
  );
};

export default OutputProjectionModule;
