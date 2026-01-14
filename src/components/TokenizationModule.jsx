import React from "react";
import { Info } from "lucide-react";
import { getTokenColor } from "../utils/transformerUtils";

const TokenizationModule = ({ inputText, setInputText, tokens, vocab }) => {
  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <p className="text-sm text-gray-700">
          The first step is to convert text into numbers. Each word (or token)
          gets mapped to a unique ID from the vocabulary.
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
                ID: {vocab.indexOf(token) !== -1 ? vocab.indexOf(token) : "?"}
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
};

export default TokenizationModule;
