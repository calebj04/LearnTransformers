import React, { useState } from "react";
import { ChevronRight, ChevronLeft, RotateCcw } from "lucide-react";
import TokenizationModule from "./components/TokenizationModule";
import EmbeddingModule from "./components/EmbeddingModule";
import PositionalEncodingModule from "./components/PositionalEncodingModule";
import AttentionModule from "./components/AttentionModule";
import FeedForwardModule from "./components/FeedForwardModule";
import OutputModule from "./components/OutputModule";

const LearnTransformers = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [inputText, setInputText] = useState("the small cat slept");
  // Removed hoveredToken, hoveredAttention, showInfo state as they are now local to components or unused

  const tokens = inputText.toLowerCase().split(" ");
  const vocab = [
    "the",
    "small",
    "cat",
    "slept",
    "on",
    "warm",
    "rug",
    "a",
    "dog",
    "ran",
  ];

  const steps = [
    {
      title: "Tokenization",
      description:
        "Converting text into numerical tokens that the model can process",
      component: (
        <TokenizationModule
          inputText={inputText}
          setInputText={setInputText}
          tokens={tokens}
          vocab={vocab}
        />
      ),
    },
    {
      title: "Token Embeddings",
      description:
        "Mapping each token to a high-dimensional vector representation",
      component: <EmbeddingModule tokens={tokens} />,
    },
    {
      title: "Positional Encodings",
      description: "Adding position information so the model knows word order",
      component: <PositionalEncodingModule tokens={tokens} />,
    },
    {
      title: "Multi-Head Attention",
      description:
        "Allowing tokens to attend to and gather information from other tokens",
      component: <AttentionModule tokens={tokens} />,
    },
    {
      title: "Feedforward Network",
      description:
        "Refining the representations through non-linear transformations",
      component: <FeedForwardModule tokens={tokens} />,
    },
    {
      title: "Output Projection",
      description:
        "Converting final representations into predictions for the next token",
      component: <OutputModule tokens={tokens} />,
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Learn Transformers
          </h1>
          <p className="text-lg text-gray-600">
            An Interactive Visual Explanation of Transformer Inference
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Based on research by Caleb Jennings, Emory University
          </p>
        </div>

        {/* Progress Bar */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, i) => (
              <div key={i} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                      i === currentStep
                        ? "bg-blue-500 text-white scale-110"
                        : i < currentStep
                        ? "bg-green-500 text-white"
                        : "bg-gray-200 text-gray-500"
                    }`}
                  >
                    {i + 1}
                  </div>
                  <div
                    className={`mt-2 text-xs font-medium text-center ${
                      i === currentStep ? "text-blue-600" : "text-gray-500"
                    }`}
                  >
                    {step.title}
                  </div>
                </div>
                {i < steps.length - 1 && (
                  <div
                    className={`h-1 flex-1 mx-2 rounded ${
                      i < currentStep ? "bg-green-500" : "bg-gray-200"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-lg shadow-xl p-8 mb-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {steps[currentStep].title}
            </h2>
            <p className="text-gray-600">{steps[currentStep].description}</p>
          </div>

          <div className="min-h-96">{steps[currentStep].component}</div>
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="flex items-center gap-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <ChevronLeft size={20} />
            Previous
          </button>

          <button
            onClick={() => setCurrentStep(0)}
            className="flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-all"
          >
            <RotateCcw size={20} />
            Reset
          </button>

          <button
            onClick={() =>
              setCurrentStep(Math.min(steps.length - 1, currentStep + 1))
            }
            disabled={currentStep === steps.length - 1}
            className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Next
            <ChevronRight size={20} />
          </button>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>
            Making Transformers Intellectually Accessible Through Visual
            Explanations
          </p>
          <p className="mt-1">CS 444: Deep Learning, Emory University</p>
        </div>
      </div>
    </div>
  );
};

export default LearnTransformers;
