import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class CandidateFitEvaluator:
    """
    Evaluates candidate fit for a job description using OpenAI API.
    """

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.model = 'gpt-3.5-turbo'
        self.daily_limit = 10000
        
        self.usage_file = 'candidate_fit_api_usage.json'
        self.usage_data = self._load_usage_data()
        
        # Cache for consistent results
        self.cache_file = 'candidate_fit_cache.json'
        self.cache_data = self._load_cache_data()

        self.prompt_template = """You are an expert HR recruiter and AI assistant. Given the following job description and candidate resume data, analyze and compare the candidate's qualifications, skills, and experience to the job requirements.

{custom_instructions}

SCORING METHODOLOGY:
- Base score starts at 0
- Add 10 points for each major required skill matched
- Add 5 points for each nice-to-have skill matched
- Add 20 points if minimum experience requirement is met
- Add 15 points if education requirement is met
- Add 5 points for each additional relevant certification/skill
- Subtract 20 points for each critical requirement missed (deal breakers)
- Subtract 10 points for each major required skill missing
- Final score should be between 0-100

EVALUATION RULES:
1. Be systematic and objective in your evaluation
2. Score each requirement individually before calculating total
3. Use consistent criteria for all evaluations
4. Consider only explicitly mentioned skills and requirements
5. Do not make assumptions about unstated qualifications

Return a JSON object with:
- "summary": A concise summary (2-4 sentences) explaining if the candidate is a good fit for the job, mentioning key matches and gaps.
- "fit_percentage": An integer percentage (0-100) representing how well the candidate fits the job. Use the scoring methodology above.
- "key_matches": Array of strings listing main skills/requirements the candidate matches.
- "key_gaps": Array of strings listing main skills/requirements the candidate lacks.
- "scoring_breakdown": Object with individual scores for transparency:
  - "required_skills_matched": number of required skills found
  - "required_skills_missed": number of required skills missing
  - "experience_match": boolean indicating if experience requirement is met
  - "education_match": boolean indicating if education requirement is met
  - "nice_to_have_matched": number of nice-to-have skills found
  - "deal_breakers_missed": number of deal breakers missing

IMPORTANT: Only return the JSON object, no extra text. Be consistent and deterministic in your scoring.

Job Description:
{job_description}

Candidate Resume:
{resume}
"""

    def _load_cache_data(self):
        """Load cache data from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception:
            return {}

    def _save_cache_data(self):
        """Save cache data to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception:
            pass

    def _generate_cache_key(self, resume_data: Dict[str, Any], job_description_data: Union[str, Dict[str, Any]], fit_options: Optional[Dict[str, Any]] = None) -> str:
        """Generate a unique cache key for the evaluation request."""
        import hashlib
        
        # Normalize inputs for consistent hashing
        if isinstance(job_description_data, dict):
            job_desc_text = json.dumps(job_description_data, sort_keys=True)
        else:
            job_desc_text = str(job_description_data)
        
        resume_text = json.dumps(resume_data, sort_keys=True)
        options_text = json.dumps(fit_options or {}, sort_keys=True)
        
        # Create hash of all inputs
        combined_text = f"{resume_text}|{job_desc_text}|{options_text}"
        return hashlib.md5(combined_text.encode()).hexdigest()

    def _load_usage_data(self):
        """Load API usage data from file."""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception:
            return {}

    def _save_usage_data(self):
        """Save API usage data to file."""
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception:
            pass

    def _get_today_key(self):
        """Get today's date as a string key."""
        return datetime.now().strftime('%Y-%m-%d')

    def _update_api_usage(self, count=1):
        """Update API usage counter."""
        today = self._get_today_key()
        if 'openai' not in self.usage_data:
            self.usage_data['openai'] = {}
        if self.model not in self.usage_data['openai']:
            self.usage_data['openai'][self.model] = {}
        if today not in self.usage_data['openai'][self.model]:
            self.usage_data['openai'][self.model][today] = 0
        self.usage_data['openai'][self.model][today] += count
        self._save_usage_data()

    def _get_today_usage(self):
        """Get today's API usage count."""
        today = self._get_today_key()
        return self.usage_data.get('openai', {}).get(self.model, {}).get(today, 0)

    def _can_use_api(self):
        """Check if API can be used based on daily limit."""
        usage = self._get_today_usage()
        return usage < self.daily_limit

    def _build_custom_instructions(self, fit_options: Optional[Dict[str, Any]]) -> str:
        """Build custom instructions based on fit_options"""
        if not fit_options:
            return ""
        
        instructions = []
        
        # Priority keywords
        if fit_options.get("priority_keywords"):
            keywords = fit_options["priority_keywords"]
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            instructions.append(f"Pay special attention to these priority keywords when evaluating fit: {keywords}")
            instructions.append(f"MANDATORY: Check each of these priority keywords [{keywords}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Required skills
        if fit_options.get("required_skills"):
            req_skills = fit_options["required_skills"]
            if isinstance(req_skills, list):
                req_skills = ", ".join(req_skills)
            instructions.append(f"Required skills: {req_skills}")
            instructions.append(f"MANDATORY: Check each required skill [{req_skills}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Minimum experience requirements
        if fit_options.get("min_experience"):
            min_exp = fit_options["min_experience"]
            instructions.append(f"The candidate must have at least {min_exp} years of relevant experience. Factor this heavily into your evaluation.")
            instructions.append(f"MANDATORY: If candidate has {min_exp}+ years experience, add 'Meets minimum experience requirement ({min_exp}+ years)' to 'key_matches'. If less, add 'Lacks minimum experience requirement ({min_exp} years)' to 'key_gaps'.")
        
        # Educational requirements
        if fit_options.get("edu_requirements"):
            edu_req = fit_options["edu_requirements"]
            instructions.append(f"Educational requirements: {edu_req}. Consider how well the candidate's education aligns with these requirements.")
            instructions.append(f"MANDATORY: Check if candidate meets education requirement '{edu_req}'. If yes, add to 'key_matches'. If no, add to 'key_gaps'.")
        
        # Weighting configuration
        weights = []
        if fit_options.get("weight_skills"):
            weights.append(f"Skills: {fit_options['weight_skills']}%")
        if fit_options.get("weight_experience"):
            weights.append(f"Experience: {fit_options['weight_experience']}%")
        if fit_options.get("weight_education"):
            weights.append(f"Education: {fit_options['weight_education']}%")
        
        if weights:
            instructions.append(f"Use the following weightage when calculating fit percentage - {', '.join(weights)}")
        
        # Deal breakers
        if fit_options.get("deal_breakers"):
            deal_breakers = fit_options["deal_breakers"]
            if isinstance(deal_breakers, list):
                deal_breakers = ", ".join(deal_breakers)
            instructions.append(f"These are deal breakers - if the candidate lacks any of these, significantly reduce the fit percentage: {deal_breakers}")
            instructions.append(f"MANDATORY: Check each deal breaker [{deal_breakers}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps' and reduce fit percentage significantly.")
        
        # Nice to have skills
        if fit_options.get("nice_to_have"):
            nice_to_have = fit_options["nice_to_have"]
            if isinstance(nice_to_have, list):
                nice_to_have = ", ".join(nice_to_have)
            instructions.append(f"These are nice-to-have skills that can boost the fit percentage: {nice_to_have}")
            instructions.append(f"MANDATORY: Check each nice-to-have skill [{nice_to_have}] against the candidate's resume. If found, add to 'key_matches' with '(nice-to-have)' notation.")
        
        # Specific experience areas
        if fit_options.get("experience_areas"):
            exp_areas = fit_options["experience_areas"]
            if isinstance(exp_areas, list):
                exp_areas = ", ".join(exp_areas)
            instructions.append(f"Required experience areas: {exp_areas}")
            instructions.append(f"MANDATORY: Check each experience area [{exp_areas}] against the candidate's background. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Technical skills
        if fit_options.get("technical_skills"):
            tech_skills = fit_options["technical_skills"]
            if isinstance(tech_skills, list):
                tech_skills = ", ".join(tech_skills)
            instructions.append(f"Required technical skills: {tech_skills}")
            instructions.append(f"MANDATORY: Check each technical skill [{tech_skills}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Certifications
        if fit_options.get("certifications"):
            certs = fit_options["certifications"]
            if isinstance(certs, list):
                certs = ", ".join(certs)
            instructions.append(f"Required/Preferred certifications: {certs}")
            instructions.append(f"MANDATORY: Check each certification [{certs}] against the candidate's credentials. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Location preferences
        if fit_options.get("location_preference"):
            location = fit_options["location_preference"]
            instructions.append(f"Location preference: {location}. Consider geographical compatibility in your evaluation.")
        
        # Salary expectations
        if fit_options.get("salary_range"):
            salary = fit_options["salary_range"]
            instructions.append(f"Salary range: {salary}. Consider if the candidate's expectations align with this range.")
        
        if instructions:
            instructions.append(
                "IMPORTANT: For every priority keyword, required skill, experience, or education requirement, "
                "explicitly list each one in either 'key_matches' (if present) or 'key_gaps' (if missing). "
                "Do not skip any. Be exhaustive."
            )
            return "EVALUATION GUIDELINES:\n" + "\n".join(f"- {instruction}" for instruction in instructions) + "\n"
        
        return ""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response text."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        return text.strip()

    def _make_openai_api_call(self, prompt: str, max_tokens: int = 1000, seed: int = 42) -> Optional[str]:
        """Make API call to OpenAI with deterministic settings."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise HR evaluation assistant. Always provide consistent, objective evaluations based solely on the provided criteria. Be deterministic in your scoring."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,  # Set to 0 for maximum determinism
                seed=seed,      # Add seed for reproducibility
                top_p=1.0,      # Use full probability distribution
                frequency_penalty=0,  # No frequency penalty
                presence_penalty=0    # No presence penalty
            )
            
            self._update_api_usage(1)
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error making OpenAI API call: {str(e)}")
            return None

    def evaluate_fit(self, resume_data: Dict[str, Any], job_description_data: Union[str, Dict[str, Any]], fit_options: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Evaluate candidate fit with optional customization parameters.
        
        Args:
            resume_data: Dictionary containing candidate resume information
            job_description_data: Job description as string or dictionary
            fit_options: Optional dictionary with evaluation parameters:
                - priority_keywords: List of important keywords to prioritize
                - required_skills: List of mandatory skills
                - min_experience: Minimum years of experience required
                - edu_requirements: Educational requirements description
                - weight_skills: Percentage weight for skills (0-100)
                - weight_experience: Percentage weight for experience (0-100)  
                - weight_education: Percentage weight for education (0-100)
                - deal_breakers: List of must-have requirements
                - nice_to_have: List of preferred but not required skills
                - experience_areas: List of required experience areas
                - technical_skills: List of required technical skills
                - certifications: List of required/preferred certifications
                - location_preference: Location requirements
                - salary_range: Expected salary range
            max_retries: Maximum number of API call retries
            
        Returns:
            Dictionary with evaluation results or None if failed
        """
        # Check cache first for consistent results
        cache_key = self._generate_cache_key(resume_data, job_description_data, fit_options)
        if cache_key in self.cache_data:
            logger.info("Returning cached result for consistent evaluation")
            return self.cache_data[cache_key]
        
        # Check if API can be used
        if not self._can_use_api():
            logger.warning(f"Daily API limit ({self.daily_limit}) reached for OpenAI")
            return None
        
        # Prepare job description text
        if isinstance(job_description_data, dict):
            job_desc_text = json.dumps(job_description_data, indent=2, ensure_ascii=False)
        else:
            job_desc_text = str(job_description_data)
        
        # Prepare resume text
        resume_text = json.dumps(resume_data, indent=2, ensure_ascii=False)
        
        # Build custom instructions based on fit_options
        custom_instructions = self._build_custom_instructions(fit_options)
        
        # Create the prompt
        prompt = self.prompt_template.format(
            custom_instructions=custom_instructions,
            job_description=job_desc_text,
            resume=resume_text
        )

        # Try API call with retries
        for attempt in range(max_retries):
            if not self._can_use_api():
                logger.warning(f"Daily API limit ({self.daily_limit}) reached for OpenAI")
                return None
            
            response_text = self._make_openai_api_call(prompt)
            
            if response_text:
                try:
                    json_str = self._extract_json(response_text)
                    parsed_result = json.loads(json_str)
                    
                    # Validate required fields
                    required_fields = ['summary', 'fit_percentage', 'key_matches', 'key_gaps']
                    if all(field in parsed_result for field in required_fields):
                        parsed_result['candidate_email'] = resume_data.get('personal_info', {}).get('email', 'N/A')
                        parsed_result['candidate_phone'] = resume_data.get('personal_info', {}).get('phone', 'N/A')
                        # Cache the result for future use
                        cache_key = self._generate_cache_key(resume_data, job_description_data, fit_options)
                        self.cache_data[cache_key] = parsed_result
                        self._save_cache_data()
                        return parsed_result
                    else:
                        logger.warning(f"Response missing required fields: {parsed_result}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    logger.warning(f"Response text: {response_text}")
            
            logger.warning(f"API call attempt {attempt + 1} failed, retrying...")
        
        logger.error("All API call attempts failed")
        return None

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics."""
        today = self._get_today_key()
        today_usage = self._get_today_usage()
        
        return {
            'today_date': today,
            'today_usage': today_usage,
            'daily_limit': self.daily_limit,
            'remaining_calls': max(0, self.daily_limit - today_usage),
            'model': self.model
        }