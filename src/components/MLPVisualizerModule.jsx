import React, { useState, useEffect, useRef, useMemo } from "react";

// Utility: generate random token embeddings
const generateRandomVector = (dim) =>
  Array.from({ length: dim }, () => Math.random() * 2 - 1);

// Linear layer: output = input * W + b
const linearLayer = (inputs, outputDim) => {
  const inputDim = inputs[0].length;
  const W = Array.from({ length: inputDim }, () =>
    Array.from({ length: outputDim }, () => Math.random() * 2 - 1)
  );
  const b = Array.from({ length: outputDim }, () => Math.random() * 2 - 1);

  const outputs = inputs.map((vec) =>
    vec.map(
      (_, colIdx) =>
        vec.reduce((sum, val, rowIdx) => sum + val * W[rowIdx][colIdx], 0) +
        b[colIdx]
    )
  );

  return { outputs, W, b };
};

// ReLU activation
const relu = (vectors) => vectors.map((vec) => vec.map((v) => Math.max(0, v)));

const MLPVisualizerModule = ({ tokens, inputTokens }) => {
  // Accept either `inputTokens` (from App) or `tokens` (alternate prop name)
  const providedTokens =
    inputTokens && inputTokens.length ? inputTokens : tokens;

  const inputDim = 6;
  const hiddenDim = 8;

  // FIX 1: Use useMemo to stabilize the token list reference
  const tokenList = useMemo(() => {
    return providedTokens && providedTokens.length
      ? providedTokens
      : [
          { word: "Hello" },
          { word: "world" },
          { word: "MLP" },
          { word: "React" },
        ];
  }, [providedTokens]);

  // FIX 2: Compute all initial data (embeddings, hidden, activated, weights) in a single useMemo call
  const { inputs, hidden, activated, weights } = useMemo(() => {
    const embeddings = tokenList.map(() => generateRandomVector(inputDim));
    const { outputs: hiddenLayer, W } = linearLayer(embeddings, hiddenDim);
    const activatedLayer = relu(hiddenLayer);

    return {
      inputs: embeddings,
      hidden: hiddenLayer,
      activated: activatedLayer,
      weights: W,
    };
  }, [tokenList, inputDim, hiddenDim]);

  const [hovered, setHovered] = useState(null);

  const containerRef = useRef(null);
  const inputRefs = useRef([]);
  // FIX 3: Create a separate ref store for the Activated layer
  const activatedRefs = useRef([]);
  const [lines, setLines] = useState([]);

  // Clear ref arrays when data changes to prevent stale elements
  useEffect(() => {
    inputRefs.current = [];
    activatedRefs.current = [];
  }, [inputs, activated]);

  // FIX 4: Dedicate one useEffect hook for layout calculation (lines)
  useEffect(() => {
    // Only run if the container and at least one ref array are populated
    if (!containerRef.current || inputRefs.current.length === 0) return;

    const computeLines = () => {
      const containerRect = containerRef.current.getBoundingClientRect();
      const newLines = [];

      // Loop through tokens (rows)
      tokenList.forEach((_, tIdx) => {
        const inputRow = inputRefs.current[tIdx] || [];
        const activatedRow = activatedRefs.current[tIdx] || [];

        // Loop through input nodes (i)
        inputRow.forEach((el1, i) => {
          if (!el1) return;
          const r1 = el1.getBoundingClientRect();
          const x1 = r1.left - containerRect.left + r1.width / 2;
          const y1 = r1.top - containerRect.top + r1.height / 2;

          // --- NEW HIGHLIGHT LOGIC CHECK ---
          const isInputNodeHovered =
            hovered &&
            hovered.layer === "input" &&
            hovered.tokenIdx === tIdx &&
            hovered.nodeIdx === i;
          // ---------------------------------

          // Loop through activated nodes (j)
          activatedRow.forEach((el2, j) => {
            if (!el2) return;
            const r2 = el2.getBoundingClientRect();
            const x2 = r2.left - containerRect.left + r2.width / 2;
            const y2 = r2.top - containerRect.top + r2.height / 2;

            let color = "rgba(0,0,0,0.12)";
            let strokeWidth = 1.5;
            let opacity = 1;

            if (weights && weights[i] && typeof weights[i][j] === "number") {
              const weight = weights[i][j];
              const baseOpacity = isInputNodeHovered ? 0.9 : 0.8;
              const opacityMagnitude = Math.min(
                Math.abs(weight) * 1.5,
                baseOpacity
              );

              // Set base color and magnitude-based opacity
              if (weight >= 0) {
                color = `rgba(220,50,50,${opacityMagnitude})`; // Red for Positive
              } else {
                color = `rgba(50,120,220,${opacityMagnitude})`; // Blue for Negative
              }
            }

            // Apply highlight styling if the input node is hovered
            if (isInputNodeHovered) {
              strokeWidth = 3;
              opacity = 1;
              // Override color for better visibility when highlighting
              const baseColor =
                weights[i][j] >= 0 ? "255,100,100" : "100,150,255";
              color = `rgba(${baseColor}, 1)`;
            } else if (!hovered) {
              // Dim lines slightly if nothing is hovered, making the highlight stand out
              opacity = 0.8;
            }

            newLines.push({ x1, y1, x2, y2, color, strokeWidth, opacity });
          });
        });
      });

      setLines(newLines);
    };

    // Use requestAnimationFrame for initial calculation to ensure layout is complete
    const rafId = requestAnimationFrame(computeLines);

    // Set up cleanup and listeners
    const handleResize = () => requestAnimationFrame(computeLines);
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", handleResize);
    };
  }, [inputs, activated, weights, tokenList, hovered]); // Dependency on 'hovered' state

  const renderNode = (val, color, tokenIdx, nodeIdx, layer, refArray) => {
    const alpha = Math.min(Math.abs(val), 1);
    const colors = {
      red: [255, 0, 0],
      green: [0, 180, 0],
      blue: [0, 120, 255],
    };
    const [r, g, b] = colors[color] || [128, 128, 128];

    return (
      <div
        key={nodeIdx}
        ref={(el) => {
          refArray.current[tokenIdx] = refArray.current[tokenIdx] || [];
          refArray.current[tokenIdx][nodeIdx] = el;
        }}
        style={{
          width: 24,
          height: 24,
          borderRadius: "50%",
          margin: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 10,
          color: "var(--text)",
          border: "1px solid rgba(0,0,0,0.08)",
          backgroundColor: `rgba(${r},${g},${b},${Math.max(alpha, 0.06)})`,
          transition: "all 0.5s ease",
          cursor: "pointer",
        }}
        onMouseEnter={() => setHovered({ layer, tokenIdx, nodeIdx, val })}
        onMouseLeave={() => setHovered(null)}
      />
    );
  };

  const renderLayer = (vectors, color, layer, refArray, showToken = true) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {vectors.map((vec, tokenIdx) => (
        <div key={tokenIdx} style={{ display: "flex", alignItems: "center" }}>
          {showToken && (
            <div style={{ width: 100, fontWeight: "bold" }}>
              {tokenList[tokenIdx]?.word}
            </div>
          )}
          <div style={{ display: "flex" }}>
            {vec.map((v, nodeIdx) =>
              renderNode(v, color, tokenIdx, nodeIdx, layer, refArray)
            )}
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h2>Feedforward MLP Layer Visualization</h2>
      <p>
        Color intensity shows value magnitude. Lines connect Input to Activated
        layer. **Hover over an input node to highlight its connections.**
      </p>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          position: "relative",
          marginTop: 30,
          minHeight: 300,
        }}
        ref={containerRef}
      >
        {/* Layer 1: Input Embeddings (Ref: inputRefs) */}
        <div style={{ flex: 1 }}>
          <h3>Input Embeddings</h3>
          {renderLayer(inputs, "red", "input", inputRefs, true)}
        </div>

        {/* Layer 2: Hidden Layer (Linear) */}
        <div style={{ flex: 1, marginLeft: 20 }}>
          <h3>Hidden Layer (Linear)</h3>
          {renderLayer(hidden, "green", "hidden", { current: [] }, false)}
        </div>

        {/* Layer 3: After ReLU Activation (Ref: activatedRefs) */}
        <div style={{ flex: 1, marginLeft: 20 }}>
          <h3>After ReLU Activation</h3>
          {renderLayer(activated, "blue", "activated", activatedRefs, false)}
        </div>

        {/* SVG Overlay */}
        <svg
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            pointerEvents: "none",
            zIndex: 10,
          }}
          width="100%"
          height="100%"
        >
          {lines.map((ln, idx) => (
            <line
              key={idx}
              x1={ln.x1}
              y1={ln.y1}
              x2={ln.x2}
              y2={ln.y2}
              stroke={ln.color}
              strokeWidth={ln.strokeWidth} // Use dynamic strokeWidth
              strokeOpacity={ln.opacity} // Use dynamic opacity
              strokeLinecap="round"
              style={{ transition: "stroke-width 0.2s, stroke-opacity 0.2s" }} // Add transition for smooth hover
            />
          ))}
        </svg>
      </div>

      {/* Hover info */}
      <div style={{ position: "relative", minHeight: 28 }}>
        <div
          style={{ position: "absolute", right: 12, bottom: 0, fontSize: 13 }}
        >
          {hovered ? (
            <span>
              Hovered **{hovered.layer}** node: **{hovered.val.toFixed(3)}**
            </span>
          ) : (
            <span style={{ color: "#666" }}>Hover a node to inspect value</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default MLPVisualizerModule;
