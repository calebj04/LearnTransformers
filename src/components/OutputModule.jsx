import React from "react";
import { Info } from "lucide-react";
import { getEmbeddingHeight } from "../utils/transformerUtils";

const OutputModule = ({ tokens }) => {
  const nextTokenProbs = [
    { token: "on", prob: 0.35 },
    { token: "peacefully", prob: 0.22 },
    { token: "soundly", prob: 0.18 },
    { token: "quietly", prob: 0.12 },
    { token: "the", prob: 0.08 },
    { token: "other", prob: 0.05 },
  ];

  return (
    <div className="space-y-6">
      <div className="bg-indigo-50 border-l-4 border-indigo-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          The final layer projects the last token's representation to a
          probability distribution over all possible next tokens.
        </p>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h4 className="font-semibold mb-4">
          Last Token: "{tokens[tokens.length - 1]}"
        </h4>

        <div className="space-y-6">
          <div>
            <div className="text-sm text-gray-600 mb-2">Final Hidden State</div>
            <div className="flex gap-1 h-24 items-end bg-gray-50 p-4 rounded">
              {[...Array(32)].map((_, dim) => (
                <div
                  key={dim}
                  className="flex-1 rounded-t bg-indigo-500"
                  style={{
                    height: `${getEmbeddingHeight(
                      tokens[tokens.length - 1],
                      dim
                    )}%`,
                    opacity: 0.7,
                  }}
                />
              ))}
            </div>
          </div>

          <div className="text-center text-2xl text-gray-400">
            ↓ Linear Projection + Softmax
          </div>

          <div>
            <div className="text-sm font-semibold mb-3">
              Next Token Predictions:
            </div>
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
          <li>
            <strong>Greedy:</strong> Always pick highest probability
          </li>
          <li>
            <strong>Top-k:</strong> Sample from top k most likely tokens
          </li>
          <li>
            <strong>Temperature:</strong> Control randomness (lower = more
            deterministic)
          </li>
        </ul>
      </div>
    </div>
  );
};

export default OutputModule;
