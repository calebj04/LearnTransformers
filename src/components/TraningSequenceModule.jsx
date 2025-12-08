import React, { useState } from "react";

// Dummy vocabulary and tokenization
const vocab = { "the": 0, "cat": 1, "sat": 2, "<pad>": 3 };
const vocabSize = Object.keys(vocab).length;

// Utility functions
const tokenize = (text) => text.toLowerCase().split(" ").map(word => vocab[word] ?? vocab["<pad>"]);
const softmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
};
const crossEntropyLoss = (predProbs, targetIdx) => -Math.log(predProbs[targetIdx] + 1e-9);

// Simulate a single Transformer block for demonstration
const transformerBlock = (embeddings) => {
  // Simple simulation: scale embeddings and add small noise
  return embeddings.map(e => e.map(val => val * 1.1 + Math.random() * 0.05));
};

// Initialize random embeddings
const initEmbeddings = (seqLen, embedDim = 8) => {
  return Array(seqLen).fill(0).map(() => Array(embedDim).fill(0).map(() => Math.random()));
};

export default function TrainingSequenceModule() {
  const [textInput, setTextInput] = useState("the cat sat");
  const [tokenIds, setTokenIds] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [logits, setLogits] = useState([]);
  const [softmaxProbs, setSoftmaxProbs] = useState([]);
  const [loss, setLoss] = useState(null);
  const [step, setStep] = useState(0);

  const embedDim = 8;

  const steps = [
    "Tokenization",
    "Embeddings",
    "Forward Pass (Transformer Blocks)",
    "Output Logits and Softmax",
    "Compute Cross-Entropy Loss",
    "Backpropagation & Parameter Update"
  ];

  const handleStepForward = () => {
    if (step >= steps.length) return;

    switch(step) {
      case 0:
        // Step 1: Tokenization
        const tokens = tokenize(textInput);
        setTokenIds(tokens);
        break;
      case 1:
        // Step 2: Initialize embeddings
        setEmbeddings(initEmbeddings(tokenIds.length, embedDim));
        break;
      case 2:
        // Step 3: Forward pass through simulated Transformer block
        setEmbeddings(prev => transformerBlock(prev));
        break;
      case 3:
        // Step 4: Output logits and softmax
        const newLogits = embeddings.map(e => e.reduce((a,b) => a+b, 0)); // sum as dummy logits
        setLogits(newLogits);
        setSoftmaxProbs(newLogits.map(l => softmax([l, 1, 1, 1])[0])); // simplified softmax
        break;
      case 4:
        // Step 5: Compute loss (cross-entropy for first token as example)
        setLoss(crossEntropyLoss(softmaxProbs, tokenIds[0]));
        break;
      case 5:
        // Step 6: Backprop & parameter update (simulated)
        setEmbeddings(prev => prev.map(e => e.map(val => val - 0.01 * val))); // simple gradient step
        break;
      default:
        break;
    }
    setStep(prev => prev + 1);
  };

  const handleReset = () => {
    setStep(0);
    setTokenIds([]);
    setEmbeddings([]);
    setLogits([]);
    setSoftmaxProbs([]);
    setLoss(null);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>Transformer Training Sequence Module</h2>
      <p>Step {step}/{steps.length}: <strong>{steps[step] || "Completed"}</strong></p>
      
      <textarea 
        rows={2} 
        cols={40} 
        value={textInput} 
        onChange={(e) => setTextInput(e.target.value)} 
      />

      <div style={{ marginTop: "10px" }}>
        <button onClick={handleStepForward} disabled={step > steps.length}>Next Step</button>
        <button onClick={handleReset} style={{ marginLeft: "10px" }}>Reset</button>
      </div>

      <div style={{ marginTop: "20px" }}>
        {step >= 1 && <div><strong>Token IDs:</strong> {tokenIds.join(", ")}</div>}
        {step >= 2 && <div><strong>Embeddings:</strong> {JSON.stringify(embeddings)}</div>}
        {step >= 3 && <div><strong>Logits:</strong> {logits.map(l => l.toFixed(2)).join(", ")}</div>}
        {step >= 3 && <div><strong>Softmax Probabilities:</strong> {softmaxProbs.map(p => p.toFixed(3)).join(", ")}</div>}
        {step >= 4 && <div><strong>Cross-Entropy Loss:</strong> {loss?.toFixed(3)}</div>}
        {step >= 5 && <div><em>Backpropagation & Parameter update simulated.</em></div>}
      </div>
    </div>
  );
}
