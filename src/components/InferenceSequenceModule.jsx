import React, { useState } from "react";

// Dummy vocabulary and tokenization
const vocab = { the: 0, cat: 1, sat: 2, "<pad>": 3, "<eos>": 4 };
const idToWord = Object.keys(vocab).reduce((obj, key) => {
  obj[vocab[key]] = key;
  return obj;
}, {});
const vocabSize = Object.keys(vocab).length;

// Utility functions
const tokenize = (text) =>
  text
    .toLowerCase()
    .split(" ")
    .map((word) => vocab[word] ?? vocab["<pad>"]);
const softmax = (logits, temperature = 1) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp((l - maxLogit) / temperature));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
};
const sampleToken = (probs, strategy = "greedy", topK = 2) => {
  if (strategy === "greedy") {
    return probs.indexOf(Math.max(...probs));
  } else if (strategy === "top-k") {
    const topIndices = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, topK)
      .map((x) => x.i);
    const topProbs = topIndices.map((i) => probs[i]);
    const total = topProbs.reduce((a, b) => a + b, 0);
    const normalized = topProbs.map((p) => p / total);
    const rand = Math.random();
    let accum = 0;
    for (let i = 0; i < normalized.length; i++) {
      accum += normalized[i];
      if (rand < accum) return topIndices[i];
    }
    return topIndices[0];
  }
  return 0;
};

// Simulated Transformer block
const transformerBlock = (embeddings) => {
  return embeddings.map((e) =>
    e.map((val) => val * 1.05 + Math.random() * 0.02)
  );
};

// Initialize random embeddings
const initEmbeddings = (seqLen, embedDim = 8) => {
  return Array(seqLen)
    .fill(0)
    .map(() =>
      Array(embedDim)
        .fill(0)
        .map(() => Math.random())
    );
};

export default function InferenceSequenceModule() {
  const [textInput, setTextInput] = useState("the");
  const [tokenIds, setTokenIds] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [logits, setLogits] = useState([]);
  const [softmaxProbs, setSoftmaxProbs] = useState([]);
  const [generatedIds, setGeneratedIds] = useState([]);
  const [step, setStep] = useState(0);
  const [samplingStrategy, setSamplingStrategy] = useState("greedy");
  const [temperature, setTemperature] = useState(1);
  const [maxLength, setMaxLength] = useState(10);

  const embedDim = 8;

  const steps = [
    "Tokenization & Embeddings",
    "Forward Pass (Transformer Blocks)",
    "Projection & Softmax",
    "Token Sampling",
    "Iterative Generation",
  ];

  const handleStepForward = () => {
    if (step >= steps.length) return;

    switch (step) {
      case 0:
        // Step 1: Tokenization and embeddings
        const tokens = tokenize(textInput);
        setTokenIds(tokens);
        setEmbeddings(initEmbeddings(tokens.length, embedDim));
        setGeneratedIds([...tokens]);
        break;
      case 1:
        // Step 2: Forward pass
        setEmbeddings((prev) => transformerBlock(prev));
        break;
      case 2:
        // Step 3: Project to logits & softmax
        const newLogits = embeddings.map((e) => e.reduce((a, b) => a + b, 0)); // sum as dummy logits
        setLogits(newLogits);
        setSoftmaxProbs(
          newLogits.map((l) => softmax([l, 1, 1, 1, 1], temperature)[0])
        ); // simplified
        break;
      case 3:
        // Step 4: Token sampling
        const sampledId = sampleToken(softmaxProbs, samplingStrategy, 2);
        setGeneratedIds((prev) => [...prev, sampledId]);
        break;
      case 4:
        // Step 5: Iterative generation until <eos> or max length
        let seq = [...generatedIds];
        while (
          seq.length < maxLength &&
          seq[seq.length - 1] !== vocab["<eos>"]
        ) {
          // simulate embeddings for last token
          let newEmb = initEmbeddings(1, embedDim);
          newEmb = transformerBlock(newEmb);
          const newLog = newEmb[0].reduce((a, b) => a + b, 0);
          const newProbs = softmax([newLog, 1, 1, 1, 1], temperature);
          const nextId = sampleToken(newProbs, samplingStrategy, 2);
          seq.push(nextId);
        }
        setGeneratedIds(seq);
        break;
      default:
        break;
    }

    setStep((prev) => prev + 1);
  };

  const handleReset = () => {
    setStep(0);
    setTokenIds([]);
    setEmbeddings([]);
    setLogits([]);
    setSoftmaxProbs([]);
    setGeneratedIds([]);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>Transformer Inference Sequence Module</h2>
      <p>
        Step {step}/{steps.length}:{" "}
        <strong>{steps[step] || "Completed"}</strong>
      </p>

      <textarea
        rows={2}
        cols={40}
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
      />

      <div style={{ marginTop: "10px" }}>
        <label>
          Sampling Strategy:
          <select
            value={samplingStrategy}
            onChange={(e) => setSamplingStrategy(e.target.value)}
          >
            <option value="greedy">Greedy</option>
            <option value="top-k">Top-K</option>
          </select>
        </label>
        <label style={{ marginLeft: "10px" }}>
          Temperature:
          <input
            type="number"
            value={temperature}
            step="0.1"
            min="0.1"
            max="5"
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </label>
        <label style={{ marginLeft: "10px" }}>
          Max Length:
          <input
            type="number"
            value={maxLength}
            min="1"
            max="20"
            onChange={(e) => setMaxLength(parseInt(e.target.value))}
          />
        </label>
      </div>

      <div style={{ marginTop: "10px" }}>
        <button onClick={handleStepForward} disabled={step > steps.length}>
          Next Step
        </button>
        <button onClick={handleReset} style={{ marginLeft: "10px" }}>
          Reset
        </button>
      </div>

      <div style={{ marginTop: "20px" }}>
        {step >= 0 && (
          <div>
            <strong>Token IDs:</strong> {tokenIds.join(", ")}
          </div>
        )}
        {step >= 0 && (
          <div>
            <strong>Embeddings:</strong> {JSON.stringify(embeddings)}
          </div>
        )}
        {step >= 2 && (
          <div>
            <strong>Logits:</strong>{" "}
            {logits.map((l) => l.toFixed(2)).join(", ")}
          </div>
        )}
        {step >= 2 && (
          <div>
            <strong>Softmax Probabilities:</strong>{" "}
            {softmaxProbs.map((p) => p.toFixed(3)).join(", ")}
          </div>
        )}
        {generatedIds.length > 0 && (
          <div>
            <strong>Generated Text:</strong>{" "}
            {generatedIds.map((id) => idToWord[id]).join(" ")}
          </div>
        )}
      </div>
    </div>
  );
}
