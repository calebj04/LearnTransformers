import React, { useState } from "react";
import { Info } from "lucide-react";
import { getTokenColor, getAttentionWeight } from "../utils/transformerUtils";

const AttentionModule = ({ tokens }) => {
  const [hoveredAttention, setHoveredAttention] = useState(null);

  return (
    <div className="space-y-6">
      <div className="bg-pink-50 border-l-4 border-pink-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          The attention mechanism lets each token "look at" other tokens to
          gather relevant context. Hover over the matrix to see attention
          strengths between tokens.
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
                const isHovered =
                  hoveredAttention?.from === fromIdx &&
                  hoveredAttention?.to === toIdx;
                return (
                  <div
                    key={toIdx}
                    className="w-16 h-16 rounded transition-all cursor-pointer border-2"
                    style={{
                      backgroundColor: `rgba(236, 72, 153, ${weight})`,
                      borderColor: isHovered
                        ? getTokenColor(fromIdx)
                        : "transparent",
                      transform: isHovered ? "scale(1.1)" : "scale(1)",
                    }}
                    onMouseEnter={() =>
                      setHoveredAttention({ from: fromIdx, to: toIdx })
                    }
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
              <span
                className="font-semibold"
                style={{ color: getTokenColor(hoveredAttention.from) }}
              >
                "{tokens[hoveredAttention.from]}"
              </span>{" "}
              attends to{" "}
              <span
                className="font-semibold"
                style={{ color: getTokenColor(hoveredAttention.to) }}
              >
                "{tokens[hoveredAttention.to]}"
              </span>{" "}
              with strength{" "}
              <span className="font-semibold">
                {getAttentionWeight(
                  hoveredAttention.from,
                  hoveredAttention.to
                ).toFixed(2)}
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
          Transformers use multiple attention heads (typically 8-12) to capture
          different types of relationships. Each head learns different attention
          patterns independently.
        </p>
      </div>
    </div>
  );
};

export default AttentionModule;
