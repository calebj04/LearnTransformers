import React from "react";
import { ChevronRight } from "lucide-react";
import { getTokenColor, getEmbeddingHeight } from "../utils/transformerUtils";

const FeedForwardModule = ({ tokens }) => {
  return (
    <div className="space-y-6">
      <div className="bg-orange-50 border-l-4 border-orange-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          After attention, each token's representation passes through a
          feedforward neural network. This refines the embeddings through
          non-linear transformations.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {tokens.map((token, i) => (
          <div key={i} className="bg-white p-6 rounded-lg shadow">
            <div
              className="font-medium mb-4"
              style={{ color: getTokenColor(i) }}
            >
              Token: "{token}"
            </div>

            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-2 text-center">
                  Input (d=512)
                </div>
                <div className="flex gap-1 h-20 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t"
                      style={{
                        backgroundColor: getTokenColor(i),
                        height: `${getEmbeddingHeight(token, dim)}%`,
                        opacity: 0.7,
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
                <div className="text-xs text-gray-600 mb-2 text-center">
                  Expanded (d=2048)
                </div>
                <div className="flex gap-0.5 h-20 items-end">
                  {[...Array(32)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t bg-orange-500"
                      style={{
                        height: `${40 + Math.random() * 60}%`,
                        opacity: 0.6,
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
                <div className="text-xs text-gray-600 mb-2 text-center">
                  Output (d=512)
                </div>
                <div className="flex gap-1 h-20 items-end">
                  {[...Array(16)].map((_, dim) => (
                    <div
                      key={dim}
                      className="flex-1 rounded-t bg-gradient-to-t from-orange-600 to-orange-400"
                      style={{
                        height: `${getEmbeddingHeight(token, dim) + 10}%`,
                        opacity: 0.8,
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
};

export default FeedForwardModule;
