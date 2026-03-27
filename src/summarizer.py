from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from src.text_preprocessor import TextPreprocessor

class Summarizer:
    @staticmethod
    def generate_general_summary(text: str, num_sentences: int = 3) -> str:
        sentences = TextPreprocessor.segment_sentences(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
            
        # TextRank approach
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Handle case where all text is stop words or empty
            return " ".join(sentences[:num_sentences])
            
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Build graph and run PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sort sentences by rank
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Extract top N sentences, maintaining original order slightly if needed, or just by rank
        # We will sort top N by their original index to keep chronological flow
        top_n = ranked_sentences[:num_sentences]
        top_sentences = [s for _, s in top_n]
        
        # Sort them back slightly by checking their index in original text
        top_sentences.sort(key=lambda s: sentences.index(s))
        
        return " ".join(top_sentences)

    @staticmethod
    def generate_risk_focused_summary(text: str, risks: list[dict]) -> str:
        # If no risks, fallback to general summary
        if not risks:
            return Summarizer.generate_general_summary(text, num_sentences=3)
            
        RISK_WEIGHTS = {
            "Liability Risk": 1.3,
            "Financial Risk": 1.2,
            "Arbitration": 1.15,
            "Data Privacy Risk": 1.05,
            "Termination & Refund Risk": 1.05
        }
        
        # 1 & 2. Weighted Risk Prioritization
        scored_risks = []
        for r in risks:
            weight = RISK_WEIGHTS.get(r['type'], 1.0)
            adj_score = r.get('severity', 0.5) * weight
            scored_risks.append({'clause': r['clause'], 'type': r['type'], 'adj_score': adj_score})
            
        # Sort by adjusted score to process highest impact first
        scored_risks.sort(key=lambda x: x['adj_score'], reverse=True)
        
        selected = []
        selected_types = set()
        vectorizer = TfidfVectorizer(stop_words='english')
        
        for sr in scored_risks:
            if len(selected) >= 3:
                break
                
            clause = sr['clause']
            ctype = sr['type']
            
            # 3. Remove Redundant Clauses
            is_redundant = False
            if selected:
                try:
                    docs = [c['clause'] for c in selected] + [clause]
                    tfidf_matrix = vectorizer.fit_transform(docs)
                    sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
                    if np.max(sims) > 0.85:
                        is_redundant = True
                except ValueError:
                    pass
                    
            if is_redundant:
                continue
                
            # 1. Diversity Check: avoid same category unless necessary
            if ctype in selected_types:
                # Check if there are other types remaining to pick from
                other_types_available = any(r['type'] not in selected_types for r in scored_risks[scored_risks.index(sr)+1:])
                if other_types_available and len(selected) < 3:
                    continue # Skip to favor diversity
                    
            selected.append(sr)
            selected_types.add(ctype)
            
        # 4. Include Context Sentence
        context_sentence = Summarizer.generate_general_summary(text, num_sentences=1).strip()
        
        final_sentences = []
        if context_sentence:
            # 5. Improve Readability
            context_sentence = context_sentence[0].upper() + context_sentence[1:]
            if not context_sentence.endswith(('.', '!', '?')):
                context_sentence += '.'
            final_sentences.append(context_sentence)
            
        for s in selected:
            clause = s['clause'].strip()
            if clause:
                # Basic cleanup
                clause = clause[0].upper() + clause[1:]
                if not clause.endswith(('.', '!', '?')):
                    clause += '.'
                # Ensure we don't accidentally add the context sentence twice if it was a risk
                if not any(clause in fs or fs in clause for fs in final_sentences):
                    final_sentences.append(clause)
                
        # Limit total sentences to max 4
        final_sentences = final_sentences[:4]
        
        # 7. Output Format: Single paragraph string
        return " ".join(final_sentences)
