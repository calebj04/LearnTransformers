import "./App.css";
import React, { useState } from "react";
import TokenizationModule from "./components/TokenizationModule.jsx";
import EmbeddingPositionalModule from "./components/EmbeddingPositionalModule.jsx";
import AttentionModule from "./components/AttentionModule.jsx";
import MLPVisualizerModule from "./components/MLPVisualizerModule.jsx";
import OutputProjectionModule from "./components/OutputProjectionModule.jsx";
import TrainingSequenceModule from "./components/TraningSequenceModule.jsx";
import InferenceSequenceModule from "./components/InferenceSequenceModule.jsx";

function App() {
  const [currentPage, setCurrentPage] = useState(0);
  const [tokens, setTokens] = useState([]); // <-- store tokens from TokenizationModule

  const pages = [
    <TokenizationModule key="tokenization" setTokens={setTokens} />,
    <EmbeddingPositionalModule key="embedding" tokens={tokens} />,
    <AttentionModule key="attention" inputText={tokens} />,
    <MLPVisualizerModule key="mlp" inputTokens={tokens} />,
    <OutputProjectionModule key="output-projection" inputTokens={tokens} />,
    <TrainingSequenceModule key="training-sequence" />,
    <InferenceSequenceModule key="inference-sequence" />,
  ];

  const next = () =>
    setCurrentPage((prev) => Math.min(prev + 1, pages.length - 1));
  const back = () => setCurrentPage((prev) => Math.max(prev - 1, 0));

  return (
    <div>
      {pages[currentPage]}
      <div className="flex justify-between mt-4 gap-2">
        <button
          onClick={back}
          disabled={currentPage === 0}
          className="px-4 py-2 rounded-md bg-gray-400 text-white disabled:opacity-50"
        >
          Back
        </button>
        <button
          onClick={next}
          disabled={currentPage === pages.length - 1}
          className="px-4 py-2 rounded-md bg-blue-600 text-white disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}

export default App;
