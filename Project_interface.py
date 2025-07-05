import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from typing import Dict, List, Any, Tuple
import logging
import os
import re
import time
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def _init_(self, post_analysis_system):
        if post_analysis_system is None:
            raise ValueError("PostAnalysisSystem instance is required")
            
        self.post_analysis_system = post_analysis_system
        self.user_patterns = [
            r"posts?\s+by\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)",
            r"([^\s]+(?:\s+[^\s]+)(?:\s,\s*[^\s]+(?:\s+[^\s]+))(?:\s+and\s+[^\s]+(?:\s+[^\s]+)*)?)'s\s+posts",
            r"first\s+post\s+(?:by|of)\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)",
            r"most\s+liked\s+posts?\s+(?:by|from)\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)"
        ]
        self.date_patterns = [
            # Title-based patterns
            r"when\s+(?:did|was)\s+(.+?)\s+post(?:ed)?",
            r"(?:date|time)\s+(?:of|for)\s+(.+?)(?:\s+post(?:ed)?)?",
            r"when\s+was\s+[\"'](.+?)[\"']\s+post(?:ed)?",
            r"post(?:ing)?\s+date\s+(?:of|for)\s+[\"'](.+?)[\"']",
            r"what\s+date\s+(?:was|is)\s+(.+?)\s+post(?:ed)?",
            
            # User-based date patterns
            r"when\s+did\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)\s+post",
            r"what\s+dates?\s+did\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)\s+post",
            r"posting\s+dates?\s+(?:for|of)\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)",
            r"when\s+has\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)\s+posted",
            r"show\s+(?:me\s+)?(?:the\s+)?dates?\s+(?:for|of)\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)'s\s+posts",
            r"on\s+what\s+dates?\s+did\s+([^\s]+(?:\s+[^\s]+)(?:\s+and\s+[^\s]+(?:\s+[^\s]+))*)\s+post"
        ]

    def _get_cached_usernames(self):
        """Get list of usernames from the PostAnalysisSystem"""
        return self.post_analysis_system.get_all_usernames()

    def _extract_usernames(self, username_str: str) -> List[str]:
        """Extract multiple usernames from a string"""
        usernames = []
        and_parts = username_str.split(' and ')
        
        for part in and_parts:
            comma_parts = part.split(',')
            for username in comma_parts:
                clean_username = username.strip()
                if clean_username:
                    usernames.append(clean_username)
        
        return usernames

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query to determine type and requirements"""
        query = query.lower().strip()
        
        result = {
            "query_type": "semantic",
            "analysis_type": "analysis",
            "length_constraint": None,
            "usernames": [],
            "post_filter": None,
            "title_search": None,
            "date_search": None,
            "original_query": query
        }
        
        # Check for date patterns first
        for pattern in self.date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_term = match.group(1).strip().strip('"\'')
                usernames = self._get_cached_usernames()
                extracted_users = self._extract_usernames(extracted_term)
                
                if any(username.lower() in extracted_term.lower() for username in usernames):
                    result["query_type"] = "user_date"
                    result["usernames"] = extracted_users
                    result["date_search"] = "user"
                else:
                    result["query_type"] = "date"
                    result["title_search"] = extracted_term
                    result["date_search"] = "title"
                return result

        # Check for length constraints
        length_patterns = {
            r"in\s+(\d+)\s+(?:lines?|words?)": "count",
            r"(\d+)\s+(?:lines?|words?)": "count"
        }
        
        for pattern, key in length_patterns.items():
            match = re.search(pattern, query)
            if match:
                result["length_constraint"] = {
                    "type": "lines" if "line" in match.group(0) else "words",
                    "count": int(match.group(1))
                }
                query = re.sub(pattern, "", query).strip()
        
        if "summary" in query:
            result["analysis_type"] = "summary"
            query = query.replace("summary", "").strip()
        elif "-analysis" in query:
            result["analysis_type"] = "analysis"
            query = query.replace("-analysis", "").strip()
        
        # Check for user-specific queries
        for pattern in self.user_patterns:
            match = re.search(pattern, query)
            if match:
                result["query_type"] = "user"
                extracted_users = self._extract_usernames(match.group(1))
                result["usernames"] = extracted_users
                
                if "first post" in query:
                    result["post_filter"] = "first"
                elif "most liked" in query:
                    result["post_filter"] = "most_liked"
                break
        
        return result

class PostAnalysisSystem:
    def _init_(self):
        """Initialize the system with ChromaDB and embedding model"""
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chromadb")
        
        try:
            self.collection = self.client.get_collection("my_collection")
        except Exception as e:
            logger.error(f"Collection not found, creating new one: {e}")
            self.collection = self.client.create_collection("my_collection")
            
        self._username_cache = {}
        self._username_map = {}
        self._refresh_caches()

    def _refresh_caches(self):
        """Refresh both username cache and map"""
        try:
            all_data = self.collection.get()
            
            self._username_cache = {}
            self._username_map = {}
            
            for i, metadata in enumerate(all_data['metadatas']):
                if metadata and 'Username' in metadata and metadata['Username']:
                    username = metadata['Username'].strip()
                    lower_username = username.lower()
                    
                    self._username_map[lower_username] = username
                    
                    if lower_username not in self._username_cache:
                        self._username_cache[lower_username] = []
                    self._username_cache[lower_username].append({
                        **metadata,
                        'document': all_data['documents'][i] if 'documents' in all_data else None
                    })
            
            logger.info(f"Refreshed caches with {len(self._username_map)} unique usernames")
        except Exception as e:
            logger.error(f"Error refreshing caches: {e}")
            raise

    def get_all_usernames(self) -> List[str]:
        """Get all unique usernames in their original case"""
        return sorted(set(self._username_map.values()))

    def _find_post_dates(self, title_search: str) -> Dict[str, Any]:
        """Find posting dates for posts matching a title"""
        try:
            results = []
            all_data = self.collection.get()
            
            for i, metadata in enumerate(all_data['metadatas']):
                if metadata and 'PostTitle' in metadata:
                    if title_search.lower() in metadata['PostTitle'].lower():
                        results.append({
                            'PostTitle': metadata['PostTitle'],
                            'Date': metadata['Date'],
                            'Username': metadata['Username']
                        })
            
            if results:
                return {
                    "type": "date_search",
                    "found": True,
                    "posts": results
                }
            return {
                "type": "not_found",
                "found": False,
                "error_message": f"No posts found matching title: {title_search}"
            }
        except Exception as e:
            return {
                "type": "error",
                "found": False,
                "error_message": f"Date search failed: {str(e)}"
            }

    def _find_user_posting_dates(self, username: str) -> Dict[str, Any]:
        """Find all posting dates for a specific user"""
        try:
            all_posts = []
            lower_username = username.lower()
            
            if lower_username in self._username_cache:
                posts = self._username_cache[lower_username]
                sorted_posts = sorted(posts, key=lambda x: x.get('Date', ''), reverse=True)
                
                for post in sorted_posts:
                    all_posts.append({
                        'PostTitle': post.get('PostTitle', 'Untitled'),
                        'Date': post.get('Date', 'Unknown'),
                        'Username': self._username_map[lower_username],
                        'Likes': post.get('Likes', '0')
                    })
                
                return {
                    "type": "user_date_search",
                    "found": True,
                    "username": self._username_map[lower_username],
                    "posts": all_posts,
                    "total_posts": len(all_posts)
                }
            
            available_usernames = ", ".join(sorted(self._username_map.values()))
            return {
                "type": "user_not_found",
                "found": False,
                "error_message": f"No posts found for user: {username}\nAvailable users: {available_usernames}"
            }
            
        except Exception as e:
            logger.error(f"User date search error: {e}")
            return {
                "type": "error",
                "found": False,
                "error_message": f"User date search failed: {str(e)}"
            }

    def search_posts(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced search function with multi-user support"""
        try:
            if not query.strip():
                return {
                    "type": "error",
                    "found": False,
                    "error_message": "Please enter a search query"
                }
            
            # Handle date queries
            if query_analysis["query_type"] == "date":
                return self._find_post_dates(query_analysis["title_search"])
            elif query_analysis["query_type"] == "user_date":
                all_user_dates = []
                for username in query_analysis["usernames"]:
                    result = self._find_user_posting_dates(username)
                    if result["found"]:
                        all_user_dates.append(result)
                
                if all_user_dates:
                    return {
                        "type": "multi_user_date_search",
                        "found": True,
                        "user_results": all_user_dates
                    }
                return {
                    "type": "user_not_found",
                    "found": False,
                    "error_message": "No posts found for the specified users"
                }

            if query_analysis["query_type"] == "user":
                all_user_posts = []
                total_posts = 0
                
                for username in query_analysis["usernames"]:
                    lower_username = username.lower()
                    
                    if lower_username in self._username_cache:
                        posts = self._username_cache[lower_username]
                        user_total_posts = len(posts)
                        total_posts += user_total_posts
                        
                        if query_analysis["post_filter"] == "first":
                            posts = sorted(posts, key=lambda x: x.get('Date', ''))[:1]
                        elif query_analysis["post_filter"] == "most_liked":
                            posts = sorted(
                                posts,
                                key=lambda x: (int(str(x.get('Likes', '0')).replace(',', '')), x.get('Date', '')),
                                reverse=True
                            )[:1]
                        else:
                            posts = posts[:min(5, user_total_posts)]
                        
                        all_user_posts.extend(posts)
                
                if all_user_posts:
                    return {
                        "type": "multi_user_posts",
                        "found": True,
                        "usernames": query_analysis["usernames"],
                        "posts": all_user_posts,
                        "total_posts": total_posts
                    }
                
                available_usernames = ", ".join(sorted(self._username_map.values()))
                return {
                    "type": "user_not_found",
                    "found": False,
                    "error_message": f"No posts found for users: {', '.join(query_analysis['usernames'])}\nAvailable users: {available_usernames}"
                }

            # Semantic search
            query_embedding = self.model.encode(query).tolist()
            return self._semantic_search(query, query_embedding)

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "type": "error",
                "found": False,
                "error_message": f"Search failed: {str(e)}"
            }

    def _semantic_search(self, query: str, query_embedding: List[float]) -> Dict[str, Any]:
        """Perform semantic search with improved relevance scoring"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results["metadatas"][0]:
                return {
                    "type": "no_content",
                    "found": False,
                    "error_message": "No relevant posts found"
                }
            
            posts = []
            for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
                similarity = 1 - distance
                if similarity > 0.3:
                    meta["similarity_score"] = round(similarity, 3)
                    posts.append(meta)
            
            if not posts:
                return {
                    "type": "no_content",
                    "found": False,
                    "error_message": "No relevant posts found"
                }
            
            return {
                "type": "content",
                "found": True,
                "query": query,
                "posts": posts,
                "query_type": "semantic_search"
            }
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            raise

def get_ollama_response(query: str, search_results: Dict[str, Any], query_analysis: Dict[str, Any]) -> str:
    """Enhanced LLM response generation with better error handling and retry logic"""
    if not search_results["found"]:
        return search_results.get("error_message", "No relevant posts found")
    
    try:
        # Format posts context
        posts_context = ""
        for i, post in enumerate(search_results["posts"][:3], 1):
            posts_context += f"\nPost {i}:\n"
            posts_context += f"Title: {post.get('PostTitle', 'Untitled')}\n"
            posts_context += f"Author: {post.get('Username', 'Unknown')}\n"
            posts_context += f"Content: {post.get('Content', '')}\n"
            posts_context += f"Likes: {post.get('Likes', 'N/A')}\n"
            posts_context += f"Date: {post.get('Date', 'N/A')}\n"
            if 'similarity_score' in post:
                posts_context += f"Relevance: {post.get('similarity_score', 'N/A')}\n"
            posts_context += "-" * 50 + "\n"

        # Build system prompt
        system_prompt = """You are a helpful expert analyst. Your task is to:
1. Analyze the provided posts thoroughly
2. Focus only on the information present in the posts
3. Provide clear, specific insights
4. Structure your response in a readable format
5. Be concise but comprehensive"""

        # Build user prompt based on query analysis
        length_instruction = ""
        if query_analysis["length_constraint"]:
            constraint = query_analysis["length_constraint"]
            length_instruction = f"Provide your response in exactly {constraint['count']} {constraint['type']}. "

        if query_analysis["query_type"] == "user":
            usernames = ", ".join(query_analysis["usernames"])
            if query_analysis["analysis_type"] == "summary":
                prompt = f"""Summarize the posts by users {usernames}. {length_instruction}
Key points to address:
1. Main topics and themes discussed
2. Key insights and perspectives
3. Overall message and expertise areas"""
            else:
                prompt = f"""Analyze these posts by users {usernames}. {length_instruction}
Please provide:
1. In-depth analysis of main topics
2. Writing style and expertise assessment
3. Notable insights and patterns
4. Development of ideas across posts"""
        else:
            if query_analysis["analysis_type"] == "summary":
                prompt = f"""Topic: {query_analysis['original_query']}

Provide a concise summary of the posts. {length_instruction}
Focus on:
1. Key findings and insights
2. Common themes
3. Important conclusions"""
            else:
                prompt = f"""Topic: {query_analysis['original_query']}

Provide a detailed analysis. {length_instruction}
Include:
1. Comprehensive overview
2. Multiple perspectives
3. Evidence from posts
4. Practical implications"""

        # Attempt to get LLM response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model="llama3.2:latest",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nAnalyze these posts:\n{posts_context}"
                        }
                    ],
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000,
                        "presence_penalty": 0.3,
                        "frequency_penalty": 0.3
                    }
                )
                
                if response and 'message' in response and 'content' in response['message']:
                    return response['message']['content']
                else:
                    raise ValueError("Invalid response format from Ollama")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Final LLM attempt failed: {str(e)}")
                    return "I apologize, but I'm having trouble generating the analysis right now. Please try again in a moment."
                else:
                    logger.warning(f"LLM attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)  # Wait before retry
                    
    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again or modify your query."

def display_search_tips():
    """Display search tips and examples in an organized way"""
    st.markdown("""
    ### 🔍 Search Tips
    
    1️⃣ *Find User Posts*
    - "posts by [username1] and [username2]"
    - "[username1], [username2]'s posts"
    - "first post by [username1] and [username2]"
    - "most liked posts by [username1], [username2]"
    
    2️⃣ *Date Queries*
    - "when did [username1] and [username2] post?"
    - "show me posting dates for [username1], [username2]"
    - "what dates did [username1] and [username2] post?"
    
    3️⃣ *Content Search*
    - Single keyword (e.g., "AI")
    - Multiple keywords (e.g., "machine learning trends")
    
    4️⃣ *Analysis Options*
    - Add "-analysis" for detailed analysis
    - Add "summary" for concise summary
    - Specify length: "in 10 lines" or "100 words"
    
    💡 *Example Queries:*
    - "AI trends -analysis"
    - "web growth summary in 10 lines"
    - "most liked posts by John and Jane"
    - "when did AI_Expert and WebGrowth post?"
    """)

def main():
    st.set_page_config(page_title="Advanced Post Analysis", layout="wide")
    
    try:
        system = PostAnalysisSystem()
        query_analyzer = QueryAnalyzer(post_analysis_system=system)
        
        st.title("QueryGo")
        
        with st.sidebar:
            st.header("📚 Available Users")
            available_usernames = system.get_all_usernames()
            st.write(", ".join(available_usernames))
            
            st.markdown("---")
            display_search_tips()
        
        user_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'AI trends -analysis' or 'posts by WebGrowth and AI_Expert'"
        )
        
        if user_query:
            with st.spinner("Searching..."):
                query_analysis = query_analyzer.parse_query(user_query)
                results = system.search_posts(user_query, query_analysis)
                
                if results["found"]:
                    if results["type"] == "date_search":
                        st.subheader("📅 Post Dates")
                        for post in results["posts"]:
                            with st.container():
                                st.markdown(f"*Title:* {post['PostTitle']}")
                                st.markdown(f"*Posted on:* {post['Date']}")
                                st.markdown(f"*By:* {post['Username']}")
                                st.markdown("---")
                    
                    elif results["type"] == "multi_user_date_search":
                        st.subheader("📅 Posting History by User")
                        for user_result in results["user_results"]:
                            st.write(f"### {user_result['username']}")
                            st.write(f"Total posts: {user_result['total_posts']}")
                            
                            for post in user_result['posts']:
                                st.markdown(f"Date: {post['Date']}")
                                st.markdown(f"Title: {post['PostTitle']}")
                                st.markdown(f"Likes: {post.get('Likes', 'N/A')}")
                                st.markdown("---")
                    
                    else:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if results["type"] == "multi_user_posts":
                                st.write(f"Posts by {', '.join(results['usernames'])} (Total: {results['total_posts']} posts)")
                            else:
                                st.write("Retrieved Posts")
                            
                            for post in results["posts"]:
                                st.markdown("---")
                                # Display each field on its own line using separate markdown calls
                                st.markdown(f"Title: {post['PostTitle']}")
                                st.markdown(f"Author: {post['Username']}")
                                st.markdown(f"Date: {post.get('Date', 'N/A')}")
                                st.markdown(f"Likes: {post.get('Likes', 'N/A')}")
                                
                                if "similarity_score" in post:
                                    st.progress(float(post['similarity_score']))
                                    st.caption(f"Relevance: {post['similarity_score']:.0%}")
                                
                                # Add content with proper spacing
                                if post.get('Content'):
                                    st.markdown("")  # Add empty line for spacing
                                    st.markdown(post['Content'])
                        
                        with col2:
                            # Display only "Analysis" or "Summary" based on query type
                            header_text = "Summary" if query_analysis["analysis_type"] == "summary" else "Analysis"
                            st.markdown(f"### {header_text}")
                            analysis = get_ollama_response(user_query, results, query_analysis)
                            st.markdown(analysis)
                else:
                    st.warning(results.get("error_message", "No results found"))
                    st.info("Check the sidebar for available usernames and search tips")
                        
    except Exception as e:
        st.error(f"System error: {str(e)}")
        st.warning("Please try refreshing the page or contact support if the issue persists.")
if __name__ == "_main_":
    main()
