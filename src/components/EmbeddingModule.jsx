import React, { useState } from "react";
import { getTokenColor, getEmbeddingHeight } from "../utils/transformerUtils";

const EmbeddingModule = ({ tokens }) => {
  const [hoveredToken, setHoveredToken] = useState(null);

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          Each token is converted into a dense vector of numbers (typically 512
          or 768 dimensions). These embeddings capture semantic meaning learned
          during training.
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
            <div
              className="font-medium mb-3"
              style={{ color: getTokenColor(i) }}
            >
              Token: "{token}"
            </div>
            <div className="flex gap-1 items-end h-24">
              {[...Array(32)].map((_, dim) => (
                <div
                  key={dim}
                  className="flex-1 rounded-t transition-all duration-300"
                  style={{
                    backgroundColor:
                      hoveredToken === i ? getTokenColor(i) : "#e5e7eb",
                    height: `${getEmbeddingHeight(token, dim)}%`,
                    opacity: hoveredToken === i ? 1 : 0.6,
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
};

export default EmbeddingModule;
