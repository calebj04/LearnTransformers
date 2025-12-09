import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Info, Play, RotateCcw } from 'lucide-react';

const LearnTransformers = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [inputText, setInputText] = useState("the small cat slept");
  const [hoveredToken, setHoveredToken] = useState(null);
  const [hoveredAttention, setHoveredAttention] = useState(null);
  const [showInfo, setShowInfo] = useState({});

  const tokens = inputText.toLowerCase().split(' ');
  const vocab = ['the', 'small', 'cat', 'slept', 'on', 'warm', 'rug', 'a', 'dog', 'ran'];
  
  const steps = [
    {
      title: "Tokenization",
      description: "Converting text into numerical tokens that the model can process"
    },
    {
      title: "Token Embeddings",
      description: "Mapping each token to a high-dimensional vector representation"
    },
    {
      title: "Positional Encodings",
      description: "Adding position information so the model knows word order"
    },
    {
      title: "Multi-Head Attention",
      description: "Allowing tokens to attend to and gather information from other tokens"
    },
    {
      title: "Feedforward Network",
      description: "Refining the representations through non-linear transformations"
    },
    {
      title: "Output Projection",
      description: "Converting final representations into predictions for the next token"
    }
  ];

  const getTokenColor = (index) => {
    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];
    return colors[index % colors.length];
  };

  const getEmbeddingHeight = (token, dim) => {
    const seed = token.charCodeAt(0) * (dim + 1);
    return 30 + (seed % 50);
  };

  const getAttentionWeight = (from, to) => {
    if (from === to) return 0.4;
    if (Math.abs(from - to) === 1) return 0.3;
    return 0.1 + Math.random() * 0.2;
  };

  const renderTokenization = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          The first step is to convert text into numbers. Each word (or token) gets mapped to a unique ID from the vocabulary.
        </p>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Input Text:</label>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
          />
        </div>

        <div className="flex flex-wrap gap-3">
          {tokens.map((token, i) => (
            <div key={i} className="text-center">
              <div 
                className="px-6 py-3 rounded-lg text-white font-medium shadow-lg transform transition-transform hover:scale-105"
                style={{ backgroundColor: getTokenColor(i) }}
              >
                {token}
              </div>
              <div className="mt-2 text-sm text-gray-600">
                ID: {vocab.indexOf(token) !== -1 ? vocab.indexOf(token) : '?'}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2 flex items-center gap-2">
          <Info size={16} />
          Vocabulary
        </h4>
        <div className="flex flex-wrap gap-2">
          {vocab.map((word, i) => (
            <span key={i} className="px-3 py-1 bg-white rounded border text-sm">
              {word} ({i})
            </span>
          ))}
        </div>
      </div>
    </div>
  );

  const renderEmbeddings = () => (
    <div className="space-y-6">
      <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          Each token is converted into a dense vector of numbers (typically 512 or 768 dimensions). 
          These embeddings capture semantic meaning learned during training.
        </p>
      </div>

      <div className="space-y-4">
        {tokens.map((token, i) => (
          <div 
            key={i}
            className="bg-white p-4 rounded-lg shadow border-l-4 transition-all hover:shadow-lg"
            style={{ borderColor: getTokenColor(i) }}
            onMouseEnter={() => setHoveredToken(i)}
            onMouseLeave={() => setHoveredToken(null)}
          >
            <div className="font-medium mb-3" style={{ color: getTokenColor(i) }}>
              Token: "{token}"
            </div>
            <div className="flex gap-1 items-end h-24">
              {[...Array(32)].map((_, dim) => (
                <div
                  key={dim}
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{
                    backgroundColor: hoveredToken === i ? getTokenColor(i) : '#e5e7eb',
                    height: `${getEmbeddingHeight(token, dim)}%`,
                    opacity: hoveredToken === i ? 1 : 0.6
                  }}
                />
              ))}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Embedding dimension: d = 512 (showing first 32 dimensions)
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderPositionalEncoding = () => (
    <div className="space-y-6">
      <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          Positional encodings add information about where each token appears in the sequence. 
          Without this, the model wouldn't know word order!
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {tokens.map((token, i) => (
          <div key={i} className="bg-white p-4 rounded-lg shadow">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium">Position {i + 1}: "{token}"</span>
              <span className="text-sm text-gray-500">Token Embedding + Positional Encoding</span>
            </div>
            
            <div className="flex gap-4 items-center">
              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-1">Token Embedding</div>
                <div className="flex gap-1 h-16 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t"
                      style={{
                        backgroundColor: getTokenColor(i),
                        height: `${getEmbeddingHeight(token, dim)}%`,
                        opacity: 0.7
                      }}
                    />
                  ))}
                </div>
              </div>

              <div className="text-2xl text-gray-400">+</div>

              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-1">Positional Encoding</div>
                <div className="flex gap-1 h-16 items-end">
                  {[...Array(16)].map((_, dim) => {
                    const height = 50 + Math.sin(i / (10000 ** (dim / 16))) * 30;
                    return (
                      <div
                        key={dim}
                        className="flex-1 rounded-t bg-green-500"
                        style={{
                          height: `${Math.abs(height)}%`,
                          opacity: 0.7
                        }}
                      />
                    );
                  })}
                </div>
              </div>

              <div className="text-2xl text-gray-400">=</div>

              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-1">Combined</div>
                <div className="flex gap-1 h-16 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t bg-gradient-to-t from-purple-500 to-blue-500"
                      style={{
                        height: `${(getEmbeddingHeight(token, dim) + Math.abs(50 + Math.sin(i / (10000 ** (dim / 16))) * 30)) / 2}%`,
                        opacity: 0.8
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAttention = () => (
    <div className="space-y-6">
      <div className="bg-pink-50 border-l-4 border-pink-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          The attention mechanism lets each token "look at" other tokens to gather relevant context. 
          Hover over the matrix to see attention strengths between tokens.
        </p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-semibold mb-4">Attention Matrix (Query × Key)</h4>
        
        <div className="inline-block">
          <div className="flex gap-2 mb-2 ml-20">
            {tokens.map((token, i) => (
              <div 
                key={i}
                className="w-16 text-center text-sm font-medium truncate"
                style={{ color: getTokenColor(i) }}
              >
                {token}
              </div>
            ))}
          </div>

          {tokens.map((fromToken, fromIdx) => (
            <div key={fromIdx} className="flex gap-2 items-center">
              <div 
                className="w-16 text-right text-sm font-medium pr-2 truncate"
                style={{ color: getTokenColor(fromIdx) }}
              >
                {fromToken}
              </div>
              {tokens.map((toToken, toIdx) => {
                const weight = getAttentionWeight(fromIdx, toIdx);
                const isHovered = hoveredAttention?.from === fromIdx && hoveredAttention?.to === toIdx;
                return (
                  <div
                    key={toIdx}
                    className="w-16 h-16 rounded transition-all cursor-pointer border-2"
                    style={{
                      backgroundColor: `rgba(236, 72, 153, ${weight})`,
                      borderColor: isHovered ? getTokenColor(fromIdx) : 'transparent',
                      transform: isHovered ? 'scale(1.1)' : 'scale(1)'
                    }}
                    onMouseEnter={() => setHoveredAttention({ from: fromIdx, to: toIdx })}
                    onMouseLeave={() => setHoveredAttention(null)}
                  >
                    {isHovered && (
                      <div className="flex items-center justify-center h-full text-xs font-bold text-white">
                        {weight.toFixed(2)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {hoveredAttention && (
          <div className="mt-4 p-3 bg-gray-50 rounded">
            <p className="text-sm">
              <span className="font-semibold" style={{ color: getTokenColor(hoveredAttention.from) }}>
                "{tokens[hoveredAttention.from]}"
              </span>
              {' '}attends to{' '}
              <span className="font-semibold" style={{ color: getTokenColor(hoveredAttention.to) }}>
                "{tokens[hoveredAttention.to]}"
              </span>
              {' '}with strength{' '}
              <span className="font-semibold">
                {getAttentionWeight(hoveredAttention.from, hoveredAttention.to).toFixed(2)}
              </span>
            </p>
          </div>
        )}
      </div>

      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold mb-2 flex items-center gap-2">
          <Info size={16} />
          Multi-Head Attention
        </h4>
        <p className="text-sm text-gray-700">
          Transformers use multiple attention heads (typically 8-12) to capture different types of relationships. 
          Each head learns different attention patterns independently.
        </p>
      </div>
    </div>
  );

  const renderFeedforward = () => (
    <div className="space-y-6">
      <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          After attention, each token's representation passes through a feedforward neural network. 
          This refines the embeddings through non-linear transformations.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {tokens.map((token, i) => (
          <div key={i} className="bg-white p-6 rounded-lg shadow">
            <div className="font-medium mb-4" style={{ color: getTokenColor(i) }}>
              Token: "{token}"
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-2 text-center">Input (d=512)</div>
                <div className="flex gap-1 h-20 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t"
                      style={{
                        backgroundColor: getTokenColor(i),
                        height: `${getEmbeddingHeight(token, dim)}%`,
                        opacity: 0.7
                      }}
                    />
                  ))}
                </div>
              </div>

              <div className="text-center">
                <div className="text-sm text-gray-600 mb-2">×W₁</div>
                <ChevronRight className="text-gray-400" />
              </div>

              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-2 text-center">Expanded (d=2048)</div>
                <div className="flex gap-0.5 h-20 items-end">
                  {[...Array(32)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t bg-orange-500"
                      style={{
                        height: `${40 + Math.random() * 60}%`,
                        opacity: 0.6
                      }}
                    />
                  ))}
                </div>
              </div>

              <div className="text-center">
                <div className="text-sm text-gray-600 mb-2">ReLU + ×W₂</div>
                <ChevronRight className="text-gray-400" />
              </div>

              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-2 text-center">Output (d=512)</div>
                <div className="flex gap-1 h-20 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t bg-gradient-to-t from-orange-600 to-orange-400"
                      style={{
                        height: `${getEmbeddingHeight(token, dim) + 10}%`,
                        opacity: 0.8
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-4 text-xs text-gray-500 text-center">
              MLP(h) = W₂ · ReLU(W₁ · h + b₁) + b₂
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderOutput = () => {
    const nextTokenProbs = [
      { token: 'on', prob: 0.35 },
      { token: 'peacefully', prob: 0.22 },
      { token: 'soundly', prob: 0.18 },
      { token: 'quietly', prob: 0.12 },
      { token: 'the', prob: 0.08 },
      { token: 'other', prob: 0.05 }
    ];

    return (
      <div className="space-y-6">
        <div className="bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded">
          <p className="text-sm text-gray-700">
            The final layer projects the last token's representation to a probability distribution over all possible next tokens.
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h4 className="font-semibold mb-4">Last Token: "{tokens[tokens.length - 1]}"</h4>
          
          <div className="space-y-6">
            <div>
              <div className="text-sm text-gray-600 mb-2">Final Hidden State</div>
              <div className="flex gap-1 h-24 items-end bg-gray-50 p-4 rounded">
                {[...Array(32)].map((_, dim) => (
                  <div
                    key={dim}
                    className="flex-1 rounded-t bg-indigo-500"
                    style={{
                      height: `${getEmbeddingHeight(tokens[tokens.length - 1], dim)}%`,
                      opacity: 0.7
                    }}
                  />
                ))}
              </div>
            </div>

            <div className="text-center text-2xl text-gray-400">↓ Linear Projection + Softmax</div>

            <div>
              <div className="text-sm font-semibold mb-3">Next Token Predictions:</div>
              <div className="space-y-2">
                {nextTokenProbs.map((item, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="w-24 text-sm font-medium">{item.token}</div>
                    <div className="flex-1">
                      <div className="h-8 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-end pr-3 text-white text-xs font-bold transition-all duration-500"
                          style={{ width: `${item.prob * 100}%` }}
                        >
                          {(item.prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-semibold mb-2 flex items-center gap-2">
            <Info size={16} />
            Sampling Strategy
          </h4>
          <p className="text-sm text-gray-700 mb-2">
            Different sampling methods select the next token:
          </p>
          <ul className="text-sm text-gray-700 space-y-1 ml-4">
            <li><strong>Greedy:</strong> Always pick highest probability</li>
            <li><strong>Top-k:</strong> Sample from top k most likely tokens</li>
            <li><strong>Temperature:</strong> Control randomness (lower = more deterministic)</li>
          </ul>
        </div>
      </div>
    );
  };

  const stepComponents = [
    renderTokenization,
    renderEmbeddings,
    renderPositionalEncoding,
    renderAttention,
    renderFeedforward,
    renderOutput
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
                        ? 'bg-blue-500 text-white scale-110'
                        : i < currentStep
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-200 text-gray-500'
                    }`}
                  >
                    {i + 1}
                  </div>
                  <div className={`mt-2 text-xs font-medium text-center ${
                    i === currentStep ? 'text-blue-600' : 'text-gray-500'
                  }`}>
                    {step.title}
                  </div>
                </div>
                {i < steps.length - 1 && (
                  <div className={`h-1 flex-1 mx-2 rounded ${
                    i < currentStep ? 'bg-green-500' : 'bg-gray-200'
                  }`} />
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

          <div className="min-h-96">
            {stepComponents[currentStep]()}
          </div>
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
            onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
            className="flex items-center gap-2 px-6 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Next
            <ChevronRight size={20} />
          </button>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>Making Transformers Intellectually Accessible Through Visual Explanations</p>
          <p className="mt-1">CS 444: Deep Learning, Emory University</p>
        </div>
      </div>
    </div>
  );
};

export default LearnTransformers;