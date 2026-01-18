import re
from typing import List, Dict, Tuple


class ResumeSemanticChunker:
    HEADLINE_PATTERNS = [
        r'^(EXPERIENCE|WORK\s*EXPERIENCE|PROFESSIONAL\s*EXPERIENCE)\b',
        r'^(EDUCATION|ACADEMIC\s*BACKGROUND|QUALIFICATIONS)\b',
        r'^(SKILLS|TECHNICAL\s*SKILLS|COMPETENCIES|CORE\s*SKILLS)\b',
        r'^(PROJECTS|PORTFOLIO|KEY\s*PROJECTS)\b',
        r'^(CERTIFICATIONS|ACHIEVEMENTS|ACCOMPLISHMENTS|AWARDS)\b',
        r'^(SUMMARY|PROFILE|ABOUT\s*ME|OBJECTIVE|PROFESSIONAL\s*SUMMARY)\b',
        r'^(CONTACT|CONTACT\s*INFO|CONTACT\s*DETAILS)\b',
        r'^([A-Z]{2,})\s*$',  
    ]
    
    HEADLINE_REGEX = re.compile('|'.join(HEADLINE_PATTERNS), re.IGNORECASE | re.MULTILINE)
    
    def __init__(self, max_chunk_tokens: int = 800, overlap: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap = overlap
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text.split())
    
    def chunk_resume(self, text: str) -> List[Dict]:
        # Layer 1: Headline patterns
        chunks = self._chunk_by_headlines(text)
        if len(chunks) > 1:
            return chunks
        
        # Layer 2: Structural detection (line length + case)
        chunks = self._chunk_by_structure(text)
        if len(chunks) > 1:
            return chunks
        
        # Layer 3: Smart fixed-size fallback
        return self._fallback_chunking(text)
    
    def _chunk_by_headlines(self, text: str) -> List[Dict]:
        headlines = []
        for match in self.HEADLINE_REGEX.finditer(text):
            headlines.append((match.start(), match.group(0).strip()))
        
        if not headlines:
            return []
        
        chunks = []
        prev_end = 0
        
        for i, (start, headline) in enumerate(headlines):
            end = headlines[i + 1][0] if i + 1 < len(headlines) else len(text)
            section_text = text[prev_end:end].strip()
            
            if section_text:
                section_chunks = self._split_long_section(section_text, headline)
                chunks.extend(section_chunks)
            
            prev_end = end
        
        return chunks
    
    def _chunk_by_structure(self, text: str) -> List[Dict]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        chunks = []
        current_section = []
        current_headline = "RESUME_OVERVIEW"
        
        for line in lines:
            # Header heuristics
            is_header = (
                len(line) < 50 and
                (line.isupper() or line.istitle()) and
                not line.endswith(('.', '!', '?')) and
                len(re.findall(r'[A-Z]', line)) / len(line) > 0.4  # High cap ratio
            )
            
            if is_header and current_section:
                section_text = ' '.join(current_section)
                if section_text.strip():
                    chunks.append({
                        'headline': current_headline,
                        'content': section_text.strip(),
                        'chunk_type': 'structure',
                        'confidence': 'medium'
                    })
                
                current_headline = line
                current_section = []
            else:
                current_section.append(line)
        
        if current_section:
            section_text = ' '.join(current_section)
            if section_text.strip():
                chunks.append({
                    'headline': current_headline,
                    'content': section_text.strip(),
                    'chunk_type': 'structure',
                    'confidence': 'medium'
                })
        
        return chunks
    
    def _split_long_section(self, section_text: str, headline: str) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', section_text.strip())
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            if current_size + sent_tokens > self.max_chunk_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'headline': headline,
                    'content': chunk_text.strip(),
                    'chunk_type': 'section',
                    'confidence': 'high'
                })
                current_chunk = current_chunk[-1:] + [sentence]
                current_size = self._estimate_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_size += sent_tokens
        
        if current_chunk:
            chunks.append({
                'headline': headline,
                'content': ' '.join(current_chunk).strip(),
                'chunk_type': 'section',
                'confidence': 'high'
            })
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            if current_size + sent_tokens > self.max_chunk_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'headline': 'CONTENT',
                    'content': chunk_text.strip(),
                    'chunk_type': 'fallback',
                    'confidence': 'low'
                })
                current_chunk = current_chunk[-1:] + [sentence]
                current_size = self._estimate_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_size += sent_tokens
        
        if current_chunk:
            chunks.append({
                'headline': 'CONTENT',
                'content': ' '.join(current_chunk).strip(),
                'chunk_type': 'fallback',
                'confidence': 'low'
            })
        
        return chunks
