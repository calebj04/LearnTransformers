import React, { useState, useEffect } from "react";

// Utility to generate random vectors
const generateRandomVector = (dim) =>
  Array.from({ length: dim }, () => Math.random());

// Compute dot-product attention scores (Q·Kᵀ)
const computeAttentionScores = (Q, K) => {
  const dim = Q[0].length;
  return Q.map((qVec) =>
    K.map(
      (kVec) =>
        qVec.reduce((sum, val, i) => sum + val * kVec[i], 0) / Math.sqrt(dim)
    )
  );
};

// Softmax rows
const softmax = (scores) => {
  return scores.map((row) => {
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sumExp);
  });
};

// Multiply attention weights with V matrix to get output
const applyAttention = (weights, V) => {
  return weights.map((row) =>
    row.map((_, colIdx) =>
      row.reduce(
        (sum, weight, tokenIdx) => sum + weight * V[tokenIdx][colIdx],
        0
      )
    )
  );
};

const AttentionModule = ({ inputText }) => {
  const [tokens, setTokens] = useState([]);
  const [head, setHead] = useState(null);
  const [hovered, setHovered] = useState(null);

  const embeddingDim = 6;

  useEffect(() => {
    if (!inputText) return;

    const tokenWords = Array.isArray(inputText)
      ? inputText.map((t) => t.word || t.text || String(t))
      : String(inputText).split(/\s+/).filter(Boolean);

    if (tokenWords.length === 0) return;

    const tokenObjs = tokenWords.map((word, idx) => ({ word, id: 101 + idx }));
    setTokens(tokenObjs);

    const Q = tokenObjs.map(() => generateRandomVector(embeddingDim));
    const K = tokenObjs.map(() => generateRandomVector(embeddingDim));
    const V = tokenObjs.map(() => generateRandomVector(embeddingDim));

    const scores = computeAttentionScores(Q, K);
    const weights = softmax(scores);
    const output = applyAttention(weights, V);

    setHead({ Q, K, V, scores, weights, output });
  }, [inputText]);

  if (!head) return null;
  const matrixStyle = (val, isHighlight = false, baseColor = "red") => {
    const baseR = baseColor === "red" ? 200 : 0;
    const baseG = baseColor === "green" ? 200 : 0;
    const baseB = baseColor === "blue" ? 200 : 0;
    const baseO = baseColor === "orange" ? 200 : 0; // Added for orange base

    let r = baseR;
    let g = baseG;
    let b = baseB;

    if (baseColor === "orange") {
        r = 255; g = 165; b = 0; // Orange color components
    }

    return {
      width: 25,
      height: 25,
      margin: 1,
      backgroundColor: `rgba(${r},${g},${b},${Math.max(
        0.08,
        Math.min(val, 1)
      )})`,
      cursor: "pointer",
      border: isHighlight ? "2px solid #222" : "1px solid rgba(0,0,0,0.06)",
      boxSizing: "border-box",
    };
  };

  // Highlight helpers: determine which rows / columns should glow when hovering
  const isQRowHighlighted = (rIdx) => {
    if (!hovered) return false;
    return (
      (hovered.type === "score" && hovered.rIdx === rIdx) ||
      (hovered.type === "weight" && hovered.rIdx === rIdx) ||
      (hovered.type === "output" && hovered.rIdx === rIdx)
    );
  };

  const isKRowHighlighted = (rIdx) => {
    if (!hovered) return false;
    return (
      (hovered.type === "score" && hovered.cIdx === rIdx) ||
      (hovered.type === "weight" && hovered.cIdx === rIdx)
    );
  };

  const isVCellHighlighted = (vRowIdx, vColIdx) => {
    // Disable highlighting for the Value (V) matrix — keep V visually stable.
    return false;
  };

  // Compute component-wise contributions q_i * k_i / sqrt(d) for hovered score
  const scoreContributions =
    hovered?.type === "score"
      ? (() => {
          const { rIdx, cIdx } = hovered;
          const q = head.Q[rIdx];
          const k = head.K[cIdx];
          const d = head.Q[0].length;
          return q.map((qv, i) => (qv * k[i]) / Math.sqrt(d));
        })()
      : null;

  const contribMax = scoreContributions
    ? Math.max(...scoreContributions.map((v) => Math.abs(v)), 1e-6)
    : 1;

  return (
    <div
      style={{
        padding: 20,
        fontFamily: "Arial, sans-serif",
        display: "flex", // Make outter div flex
        flexDirection: "column", // Stack items vertically
        alignItems: "center", // Center items horizontally
      }}
    >
      <h2>Attention Module</h2>
      <p>
        Hover over Q·Kᵀ, attention, or output elements to highlight
        contributions.
      </p>

      {/* Q, K and V matrices */}
      <div
        style={{
          display: "flex",
          gap: 40,
          marginBottom: 20,
          justifyContent: "center", // Center the inner elements horizontally
          width: "100%", // Take up full width for centering
        }}
      >
        {/* Q */}
        <div>
          <p>
            <strong>Query (Q):</strong>
          </p>
          {head.Q.map((row, rIdx) => (
            <div key={rIdx} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ width: 60, marginRight: 5 }}>
                {tokens[rIdx].word}
              </div>
              {row.map((val, cIdx) => (
                <div
                  key={cIdx}
                  style={matrixStyle(val, isQRowHighlighted(rIdx), "red")}
                />
              ))}
            </div>
          ))}
        </div>

        {/* K */}
        <div>
          <p>
            <strong>Key (K):</strong>
          </p>
          {head.K.map((row, rIdx) => (
            <div key={rIdx} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ width: 60, marginRight: 5 }}>
                {tokens[rIdx].word}
              </div>
              {row.map((val, cIdx) => (
                <div
                  key={cIdx}
                  style={matrixStyle(val, isKRowHighlighted(rIdx), "green")}
                />
              ))}
            </div>
          ))}
        </div>

        {/* V */}
        <div>
          <p>
            <strong>Value (V):</strong>
          </p>
          {head.V.map((row, rIdx) => (
            <div key={rIdx} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ width: 60, marginRight: 5 }}>
                {tokens[rIdx].word}
              </div>
              {row.map((val, cIdx) => (
                <div
                  key={cIdx}
                  style={matrixStyle(
                    val,
                    isVCellHighlighted(rIdx, cIdx),
                    "blue"
                  )}
                />
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Visual equation: Q x K^T -> A */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontWeight: 600 }}>Flow:</div>
          <div
            style={{
              background: "#fafafa",
              padding: "6px 10px",
              borderRadius: 6,
            }}
          >
            Q × Kᵀ / √d_k → softmax → A
          </div>
          <div style={{ color: "#666", fontSize: 13 }}>
            (A is the attention matrix used to compute Output = A × V)
          </div>
        </div>
      </div>

      {/* Q·Kᵀ */}
      <div
        style={{
          marginBottom: 20,
          display: "flex", // Make this section container flex
          flexDirection: "column", // Stack items vertically
          alignItems: "center", // Center items horizontally
          width: "100%", // Take up full width for centering
        }}
      >
        <p>
          <strong>Q·Kᵀ Scores:</strong>
        </p>
        {head.scores.map((row, rIdx) => (
          <div key={rIdx} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ width: 60, marginRight: 5 }}>{tokens[rIdx].word}</div>
            {row.map((val, cIdx) => {
              const isHighlight =
                hovered?.type === "score" &&
                hovered.rIdx === rIdx &&
                hovered.cIdx === cIdx;
              return (
                <div
                  key={cIdx}
                  style={matrixStyle(val, isHighlight)}
                  onMouseEnter={() => setHovered({ type: "score", rIdx, cIdx })}
                  onMouseLeave={() => setHovered(null)}
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Component-wise contributions (reserved area to avoid layout shifts) */}
      <div
        style={{
          marginBottom: 18,
          minHeight: 120,
          position: "relative",
          display: "flex", // Make this section container flex
          flexDirection: "column", // Stack items vertically
          alignItems: "center", // Center content horizontally
          width: "100%", // Take up full width for centering
        }}
      >
        <div style={{ position: "absolute", left: 0, right: 0, top: 0 }}>
          <p style={{ textAlign: "center" }}>
            <strong>Component-wise contributions (q_i · k_i / √d):</strong>
          </p>
        </div>

        {/* Render bars only when scoreContributions exists; kept absolutely positioned to prevent reflow */}
        {scoreContributions && (
          <div style={{ position: "absolute", left: 0, right: 0, top: 28 }}>
            <div
              style={{
                display: "flex",
                gap: 12,
                alignItems: "flex-end",
                justifyContent: "center", // Center the bar visualization
              }}
            >
              {scoreContributions.map((v, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                  }}
                >
                  <div
                    style={{
                      width: 30,
                      height: 48,
                      display: "flex",
                      alignItems: "flex-end",
                      background: "#fff",
                      borderRadius: 4,
                      overflow: "hidden",
                      boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.03)",
                    }}
                  >
                    <div
                      style={{
                        width: "100%",
                        height: `${(Math.abs(v) / contribMax) * 100}%`,
                        background: v >= 0 ? "#4f83cc" : "#cc4f4f",
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 11, marginTop: 6 }}>
                    {v.toFixed(2)}
                  </div>
                  <div style={{ fontSize: 11, color: "#666" }}>{`d${i}`}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Attention weights */}
      <div
        style={{
          marginBottom: 20,
          display: "flex", // Make this section container flex
          flexDirection: "column", // Stack items vertically
          alignItems: "center", // Center items horizontally
          width: "100%", // Take up full width for centering
        }}
      >
        <p>
          <strong>Attention Weights (softmaxed):</strong>
        </p>
        {head.weights.map((row, rIdx) => (
          <div key={rIdx} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ width: 60, marginRight: 5 }}>{tokens[rIdx].word}</div>
            {row.map((val, cIdx) => {
              const isExact =
                hovered?.type === "weight" &&
                hovered.rIdx === rIdx &&
                hovered.cIdx === cIdx;
              const isRowHighlight =
                hovered?.type === "output" && hovered.rIdx === rIdx;
              const isHighlight = isExact || isRowHighlight;
              return (
                <div
                  key={cIdx}
                  style={matrixStyle(val, isHighlight, "orange")}
                  onMouseEnter={() =>
                    setHovered({ type: "weight", rIdx, cIdx })
                  }
                  onMouseLeave={() => setHovered(null)}
                />
              );
            })}
          </div>
        ))}
      </div>

      {/* Output */}
      <div
        style={{
          display: "flex", // Make this section container flex
          flexDirection: "column", // Stack items vertically
          alignItems: "center", // Center items horizontally
          width: "100%", // Take up full width for centering
        }}
      >
        <p>
          <strong>Output = Attention·V:</strong>
        </p>
        <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
          {head.output.map((row, rIdx) => (
            <div key={rIdx}>
              <div style={{ textAlign: "center" }}>{tokens[rIdx].word}</div>
              {row.map((val, cIdx) => {
                const isExactOutputHighlight =
                  hovered?.type === "output" &&
                  hovered.rIdx === rIdx &&
                  hovered.cIdx === cIdx;
                const isRowFromWeightHover =
                  hovered?.type === "weight" && hovered.rIdx === rIdx;
                const highlight =
                  isExactOutputHighlight || isRowFromWeightHover;
                return (
                  <div
                    key={cIdx}
                    style={{
                      width: 25,
                      height: 25,
                      margin: 1,
                      backgroundColor: highlight
                        ? "purple"
                        : `rgba(128,0,128,${val})`,
                      border: isExactOutputHighlight
                        ? "2px solid #111"
                        : undefined,
                      boxSizing: "border-box",
                    }}
                    onMouseEnter={() =>
                      setHovered({ type: "output", rIdx, cIdx })
                    }
                    onMouseLeave={() => setHovered(null)}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AttentionModule;