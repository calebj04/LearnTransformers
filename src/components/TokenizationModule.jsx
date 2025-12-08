import React, { useState, useRef } from "react";

// Mock tokenizer for demonstration purposes
const mockTokenizer = (text) => {
  // Example mapping: split by spaces and assign token IDs (fake vocabulary)
  const words = text.split(" ");
  const vocab = {};
  let nextId = 101;
  return words.map((word) => {
    if (!(word in vocab)) {
      vocab[word] = nextId++;
    }
    return { word, id: vocab[word] };
  });
};

const TokenizationModule = ({ setTokens: setTokensProp }) => {
  const [inputText, setInputText] = useState(
    "The quick brown fox jumps over the lazy dog"
  );
  const [tokens, localSetTokens] = useState(mockTokenizer(inputText));
  const [hoveredWord, setHoveredWord] = useState(null);

  const handleTokenize = () => {
    const toks = mockTokenizer(inputText);
    localSetTokens(toks);
    if (typeof setTokensProp === "function") setTokensProp(toks);
  };
  const clearHoverRef = useRef(null);

  return (
    <div className="p-6 font-sans">
      <h2 className="text-2xl font-semibold mb-2">4.2.1 Tokenization Module</h2>
      <p className="text-white mb-4">
        Before feeding text into a transformer model, it must be converted into
        a sequence of token IDs using a subword tokenizer. This component
        visualizes that process by mapping user-provided text to discrete tokens
        and displaying the vocabulary index for each token.
      </p>

      <div className="mb-4">
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={3}
          className="w-full p-3 text-base rounded-md border border-gray-200"
        />
        <button
          onClick={handleTokenize}
          className="mt-3 px-4 py-2 text-base rounded-md bg-blue-600 text-white hover:bg-blue-700"
        >
          Tokenize
        </button>
      </div>

      <div className="p-4 bg-gray-100 rounded-lg mt-5">
        <p className="font-medium text-black">Tokenized Output:</p>
        <div className="flex flex-wrap mt-3 justify-center">
          {tokens.map((token, index) => (
            <span
              key={index}
              tabIndex={0}
              onMouseEnter={() => {
                if (clearHoverRef.current) {
                  clearTimeout(clearHoverRef.current);
                  clearHoverRef.current = null;
                }
                setHoveredWord(token.word);
              }}
              onMouseLeave={() => {
                clearHoverRef.current = setTimeout(
                  () => setHoveredWord(null),
                  120
                );
              }}
              onFocus={() => {
                if (clearHoverRef.current) {
                  clearTimeout(clearHoverRef.current);
                  clearHoverRef.current = null;
                }
                setHoveredWord(token.word);
              }}
              onBlur={() => {
                clearHoverRef.current = setTimeout(
                  () => setHoveredWord(null),
                  120
                );
              }}
              className={`mr-2 mb-2 px-3 py-1 rounded-md cursor-pointer transition-colors duration-150 ${
                hoveredWord === token.word
                  ? "bg-slate-800 text-white"
                  : "bg-gray-200 text-black"
              }`}
            >
              ID: {token.id}
            </span>
          ))}
        </div>
        <div className="mt-3 relative" style={{ minHeight: 32 }}>
          {/* Reserve vertical space so showing hover text doesn't reflow layout */}
          <div
            onMouseEnter={() => {
              if (clearHoverRef.current) {
                clearTimeout(clearHoverRef.current);
                clearHoverRef.current = null;
              }
            }}
            onMouseLeave={() => {
              clearHoverRef.current = setTimeout(
                () => setHoveredWord(null),
                120
              );
            }}
            className="absolute left-0 right-0 top-0"
            style={{ display: "flex", alignItems: "center", minHeight: 32 }}
          >
            {hoveredWord ? (
              <p className="mt-0 font-semibold text-slate-800">
                Original token: "{hoveredWord}"
              </p>
            ) : (
              <p className="mt-0 text-gray-600">
                Hover a token to see the original text
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TokenizationModule;
