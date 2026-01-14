import React from "react";
import { getTokenColor, getEmbeddingHeight } from "../utils/transformerUtils";

const PositionalEncodingModule = ({ tokens }) => {
  return (
    <div className="space-y-6">
      <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          Positional encodings add information about where each token appears in
          the sequence. Without this, the model wouldn't know word order!
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {tokens.map((token, i) => (
          <div key={i} className="bg-white p-4 rounded-lg shadow">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium">
                Position {i + 1}: "{token}"
              </span>
              <span className="text-sm text-gray-500">
                Token Embedding + Positional Encoding
              </span>
            </div>

            <div className="flex gap-4 items-center">
              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-1">
                  Token Embedding
                </div>
                <div className="flex gap-1 h-16 items-end">
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

              <div className="text-2xl text-gray-400">+</div>

              <div className="flex-1">
                <div className="text-xs text-gray-600 mb-1">
                  Positional Encoding
                </div>
                <div className="flex gap-1 h-16 items-end">
                  {[...Array(16)].map((_, dim) => {
                    const height = 50 + Math.sin(i / 10000 ** (dim / 16)) * 30;
                    return (
                      <div
                        key={dim}
                        className="flex-1 rounded-t bg-green-500"
                        style={{
                          height: `${Math.abs(height)}%`,
                          opacity: 0.7,
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
                        height: `${
                          (getEmbeddingHeight(token, dim) +
                            Math.abs(
                              50 + Math.sin(i / 10000 ** (dim / 16)) * 30
                            )) /
                          2
                        }%`,
                        opacity: 0.8,
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
};

export default PositionalEncodingModule;
