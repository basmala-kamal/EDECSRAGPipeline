from typing import List
import re


class TextChunker:
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Simple approximation: split on whitespace
        return len(text.split())

    def chunk_text(self, text: str) -> List[str]:
        # Split into sentences for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If single sentence exceeds chunk_size, split it by words
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large sentence into word-based chunks
                words = sentence.split()
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    word_chunk = words[i:i + self.chunk_size]
                    chunks.append(' '.join(word_chunk))
                continue

            # Add sentence to current chunk
            if current_size + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_tokens
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap_chunk = []
                overlap_size = 0

                # Add sentences from the end for overlap
                for sent in reversed(current_chunk):
                    sent_tokens = self._estimate_tokens(sent)
                    if overlap_size + sent_tokens <= self.overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_size += sent_tokens
                    else:
                        break

                # Start new chunk with overlap + current sentence
                current_chunk = overlap_chunk + [sentence]
                current_size = overlap_size + sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return [chunk.strip() for chunk in chunks if chunk.strip()]
