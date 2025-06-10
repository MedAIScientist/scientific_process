import re
import json
from typing import Dict, List, Any, Tuple


class HypothesisPipeline:
    """
    A class to extract and process hypotheses from Gemini API responses
    for further use in a scientific paper generation pipeline.
    """

    def __init__(self):
        self.hypotheses = []
        self.rationales = []
        self.summary = ""

    def extract_from_gemini_response(self, response_text: str) -> bool:
        """
        Extract hypotheses, rationales, and summary from Gemini API response.

        Args:
            response_text: The text response from Gemini API

        Returns:
            bool: True if extraction was successful, False otherwise
        """
        try:
            print(f"[DEBUG] Processing response of {len(response_text)} characters")
            
            # Extract summary (case insensitive)
            summary_match = re.search(r'###\s*[Ss]ummary\s*\n(.*?)(?=###|\Z)', response_text, re.DOTALL)
            if summary_match:
                self.summary = summary_match.group(1).strip()
                print(f"[DEBUG] Summary extracted: {len(self.summary)} characters")
            else:
                print("[DEBUG] No summary found")

            # Extract hypotheses and rationales (case insensitive)
            hyp_pattern = r'###\s*[Hh]ypothesis\s*(\d+)\s*\n(.*?)(?:\*\*Rationale:\*\*|\*\*Rationale\*\*:)(.*?)(?=###|\Z)'
            hypotheses_matches = re.findall(hyp_pattern, response_text, re.DOTALL)

            print(f"[DEBUG] Found {len(hypotheses_matches)} hypothesis matches")

            self.hypotheses = []
            self.rationales = []

            for i, (num, hypothesis, rationale) in enumerate(hypotheses_matches):
                clean_hypothesis = hypothesis.strip()
                clean_rationale = rationale.strip()
                self.hypotheses.append(clean_hypothesis)
                self.rationales.append(clean_rationale)
                print(f"[DEBUG] Hypothesis {i+1}: {clean_hypothesis[:50]}...")

            print(f"[DEBUG] Total extracted: {len(self.hypotheses)} hypotheses")
            return len(self.hypotheses) > 0

        except Exception as e:
            print(f"[DEBUG] Error extracting hypotheses: {str(e)}")
            return False

    def get_hypotheses_with_rationales(self) -> List[Dict[str, str]]:
        """
        Return hypotheses with their rationales as a list of dictionaries.

        Returns:
            List of dictionaries with 'hypothesis' and 'rationale' keys
        """
        return [
            {"hypothesis": h, "rationale": r}
            for h, r in zip(self.hypotheses, self.rationales)
        ]

    def get_summary(self) -> str:
        """Return the dataset summary."""
        return self.summary

    def export_to_json(self, filepath: str) -> bool:
        """
        Export hypotheses, rationales, and summary to a JSON file.

        Args:
            filepath: Path to save the JSON file

        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            data = {
                "summary": self.summary,
                "hypotheses": [
                    {"id": i + 1, "hypothesis": h, "rationale": r}
                    for i, (h, r) in enumerate(zip(self.hypotheses, self.rationales))
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
            return False

    def import_from_json(self, filepath: str) -> bool:
        """
        Import hypotheses, rationales, and summary from a JSON file.

        Args:
            filepath: Path to read the JSON file from

        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.summary = data.get("summary", "")

            self.hypotheses = []
            self.rationales = []

            for item in data.get("hypotheses", []):
                self.hypotheses.append(item.get("hypothesis", ""))
                self.rationales.append(item.get("rationale", ""))

            return True

        except Exception as e:
            print(f"Error importing from JSON: {str(e)}")
            return False


# Example of how to use this in your Streamlit app:
def integrate_with_streamlit(gemini_response: str) -> Dict[str, Any]:
    """
    Process Gemini response and return structured data for further pipeline steps.

    Args:
        gemini_response: Text response from Gemini API

    Returns:
        Dictionary with extracted information
    """
    pipeline = HypothesisPipeline()
    success = pipeline.extract_from_gemini_response(gemini_response)

    if success:
        return {
            "summary": pipeline.get_summary(),
            "hypotheses": pipeline.get_hypotheses_with_rationales(),
            "status": "success"
        }
    else:
        return {
            "status": "error",
            "message": "Failed to extract hypotheses from response"
        }


# Example usage outside Streamlit:
if __name__ == "__main__":
    # Example Gemini response
    example_response = """
    ### Summary
    This dataset contains information about customer behavior including demographics, purchase history, and satisfaction ratings across different product categories and time periods.

    ### hypothesis 1
    Customers with higher income levels show significantly greater satisfaction with premium products compared to budget products.
    **Rationale:** The data shows a strong positive correlation between income level and satisfaction ratings for premium product categories, suggesting that higher-income customers may have different expectations or experiences with premium offerings.

    ### hypothesis 2
    Seasonal variations significantly impact purchase frequency across all product categories, with distinct patterns for each product type.
    **Rationale:** Looking at the timestamp data and purchase frequency, there appear to be cyclical patterns in buying behavior that differ by product category, indicating potential seasonal effects on consumer preferences.

    ### hypothesis 3
    Customer age is a stronger predictor of product loyalty than length of customer relationship.
    **Rationale:** The data shows that while both age and customer tenure have positive correlations with repeat purchases, the correlation coefficient is notably stronger for age (0.67) than for relationship length (0.42), suggesting age may be more influential in determining loyalty behaviors.
    """

    pipeline = HypothesisPipeline()
    pipeline.extract_from_gemini_response(example_response)

    # Save to file
    pipeline.export_to_json("hypotheses_output.json")

    # Print extracted data
    print("Summary:", pipeline.get_summary())
    print("\nHypotheses with Rationales:")
    for i, item in enumerate(pipeline.get_hypotheses_with_rationales()):
        print(f"\nhypothesis {i + 1}:")
        print(f"Statement: {item['hypothesis']}")
        print(f"Rationale: {item['rationale']}")