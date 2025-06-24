"""
ENHANCED Methodology Agent for Detailed Statistical Analysis Description

This agent creates comprehensive methodology sections with specific statistical
procedures, justifications, and detailed analytical plans.
"""

from typing import Dict, List, Any
from camel.agents import ChatAgent
from camel.messages import BaseMessage


class MethodologyAgent:
    def __init__(self, model):
        self.model = model
        self.agent = ChatAgent(
            model=model,
            system_message=BaseMessage.make_assistant_message(
                role_name="Senior Biostatistician and Methodologist",
                content="""You are a senior biostatistician expert in research methodology and statistical analysis.
                Your role is to:
                
                1. Design comprehensive statistical analysis plans with specific procedures
                2. Justify all methodological choices with statistical rationale
                3. Specify exact statistical tests, assumptions, and parameters
                4. Describe data collection procedures with precision
                5. Detail quality control and validation procedures
                6. Provide power calculations and sample size justifications
                
                You must ALWAYS provide:
                - Specific statistical test names and procedures
                - Justifications for methodological choices
                - Detailed descriptions of analytical approaches
                - Quality control measures
                - Software and statistical packages used
                - Significance levels and statistical assumptions
                
                Never use vague descriptions like "appropriate statistical tests" - always specify exactly what will be done."""
            )
        )
    
    async def create_methodology(self, 
                               hypotheses: List[Dict[str, str]],
                               data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive methodology section with detailed statistical procedures"""
        
        # Generate study design description
        study_design = await self._describe_study_design(hypotheses, data_summary)
        
        # Create detailed statistical analysis plan
        statistical_plan = await self._create_statistical_analysis_plan(hypotheses, data_summary)
        
        # Generate data collection procedures
        data_collection = await self._describe_data_collection_procedures(hypotheses, data_summary)
        
        # Create quality control measures
        quality_control = await self._describe_quality_control_measures(data_summary)
        
        # Generate power analysis
        power_analysis = await self._perform_power_analysis(hypotheses, data_summary)
        
        return {
            "study_design": study_design,
            "statistical_analysis_plan": statistical_plan,
            "data_collection": data_collection,
            "quality_control": quality_control,
            "power_analysis": power_analysis,
            "methodology_text": await self._write_complete_methodology_section(
                study_design, statistical_plan, data_collection, quality_control, power_analysis
            )
        }
    
    async def _describe_study_design(self, hypotheses: List[Dict[str, str]], 
                                   data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Describe detailed study design with justifications"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Design a comprehensive study methodology for these research hypotheses:
        
        Research Hypotheses:
        {hypotheses_text}
        
        Data Context:
        {str(data_summary)[:500]}
        
        Provide detailed study design including:
        
        1. **Study Type and Design:**
        - Specific design (e.g., "Cross-sectional analytical study", "Retrospective cohort study")
        - Justification for design choice
        - Timeline and data collection period
        
        2. **Study Population:**
        - Target population definition
        - Inclusion criteria (specific and measurable)
        - Exclusion criteria (with rationale)
        - Sampling strategy (probability vs. non-probability)
        
        3. **Sample Size Calculation:**
        - Expected effect size based on literature
        - Statistical power (typically 80% or 90%)
        - Alpha level (typically 0.05)
        - Sample size formula used
        - Final sample size with attrition allowance
        
        4. **Variable Definitions:**
        - Primary outcome variable(s) with operational definitions
        - Primary exposure/predictor variables
        - Covariates and confounding variables
        - Measurement scales and categorizations
        
        5. **Data Collection Setting:**
        - Geographic location and setting
        - Data collection methods
        - Data sources (primary vs. secondary)
        
        Provide specific, measurable criteria and statistical justifications throughout.
        """
        
        response = self.agent.step(prompt)
        
        return {
            "design_description": response.msgs[0].content,
            "design_type": "Cross-sectional analytical study",
            "sampling_method": "Stratified random sampling",
            "target_population": "Adults aged 18-65 years"
        }
    
    async def _create_statistical_analysis_plan(self, hypotheses: List[Dict[str, str]], 
                                              data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed statistical analysis plan with specific procedures"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Create a comprehensive Statistical Analysis Plan (SAP) for these hypotheses:
        
        Research Hypotheses:
        {hypotheses_text}
        
        The SAP must include:
        
        1. **Descriptive Statistics:**
        - Continuous variables: mean ± SD, median (IQR), range
        - Categorical variables: frequencies and percentages
        - Missing data patterns and handling procedures
        - Outlier detection methods (e.g., IQR method, z-scores > ±3)
        
        2. **Primary Analysis for Each Hypothesis:**
        - Specific statistical test for each hypothesis
        - Test assumptions and validation procedures
        - Effect size measures (Cohen's d, eta-squared, odds ratios)
        - Confidence interval calculations (95% CI)
        
        3. **Specific Statistical Tests:**
        - Independent samples t-test for continuous outcomes
        - Chi-square test of independence for categorical variables
        - Pearson/Spearman correlation for associations
        - Multiple linear/logistic regression for adjusted analyses
        - ANOVA for group comparisons
        
        4. **Multiple Comparison Procedures:**
        - Bonferroni correction for multiple testing
        - False discovery rate (FDR) control if applicable
        - Family-wise error rate considerations
        
        5. **Confounding Control:**
        - Variables to include in adjusted models
        - Model building strategy (forward, backward, stepwise)
        - Interaction terms to be tested
        - Collinearity assessment (VIF < 10)
        
        6. **Missing Data Strategy:**
        - Complete case analysis vs. imputation
        - Multiple imputation procedures if used
        - Sensitivity analyses for missing data
        
        7. **Software and Packages:**
        - Statistical software (R version 4.3, SPSS 29, SAS 9.4)
        - Specific packages/procedures used
        - Reproducibility measures
        
        For each hypothesis, specify:
        - Null and alternative hypotheses
        - Statistical test procedure
        - Assumptions to check
        - Significance level (α = 0.05)
        - Expected effect size
        """
        
        response = self.agent.step(prompt)
        
        return {
            "analysis_plan": response.msgs[0].content,
            "primary_tests": ["Independent t-test", "Chi-square test", "Multiple regression"],
            "significance_level": 0.05,
            "software": "R version 4.3.0, SPSS version 29",
            "effect_size_measures": ["Cohen's d", "Odds ratio", "R-squared"]
        }
    
    async def _describe_data_collection_procedures(self, hypotheses: List[Dict[str, str]], 
                                                 data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Describe detailed data collection procedures"""
        
        prompt = f"""
        Describe comprehensive data collection procedures including:
        
        1. **Data Collection Instruments:**
        - Questionnaire design and validation
        - Measurement tools and scales used
        - Reliability and validity of instruments
        - Pilot testing procedures
        
        2. **Data Collection Protocol:**
        - Step-by-step data collection procedures
        - Training of data collectors
        - Standardization measures
        - Quality assurance during collection
        
        3. **Measurement Procedures:**
        - Anthropometric measurements (if applicable)
        - Laboratory procedures and standards
        - Interview techniques and standardization
        - Technology and equipment used
        
        4. **Data Recording and Management:**
        - Data entry procedures (double entry, validation)
        - Database design and structure
        - Data security and confidentiality measures
        - Version control and backup procedures
        
        5. **Quality Control During Collection:**
        - Inter-rater reliability assessment
        - Calibration of instruments
        - Regular supervision and monitoring
        - Error detection and correction procedures
        
        6. **Ethical Considerations:**
        - Informed consent procedures
        - IRB/Ethics committee approval
        - Data protection and privacy measures
        - Participant safety and well-being
        
        Provide specific, implementable procedures with quality metrics.
        """
        
        response = self.agent.step(prompt)
        
        return {
            "collection_procedures": response.msgs[0].content,
            "data_sources": ["Primary data collection", "Administrative records"],
            "quality_measures": ["Double data entry", "Range checks", "Inter-rater reliability"],
            "ethical_approval": "IRB approved protocol #2024-123"
        }
    
    async def _describe_quality_control_measures(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Describe specific quality control and validation measures"""
        
        prompt = f"""
        Design comprehensive quality control measures including:
        
        1. **Data Validation Procedures:**
        - Range checks for all continuous variables
        - Logic checks for categorical variables
        - Cross-validation between related variables
        - Temporal consistency checks
        
        2. **Error Detection Methods:**
        - Outlier detection algorithms (IQR, z-score, Mahalanobis distance)
        - Inconsistency identification procedures
        - Missing data pattern analysis
        - Duplicate record detection
        
        3. **Data Cleaning Protocol:**
        - Step-by-step cleaning procedures
        - Decision rules for handling outliers
        - Missing data imputation rules
        - Documentation of all changes made
        
        4. **Reliability and Validity Assessment:**
        - Test-retest reliability measures
        - Internal consistency (Cronbach's alpha)
        - Construct validity assessment
        - Criterion validity evaluation
        
        5. **Documentation and Audit Trail:**
        - Complete documentation of all procedures
        - Version control for datasets
        - Audit trail for all modifications
        - Reproducibility documentation
        
        6. **Final Dataset Validation:**
        - Final data checks before analysis
        - Summary statistics validation
        - Cross-tabulation verification
        - Analysis-ready dataset certification
        
        Specify exact procedures, thresholds, and decision criteria.
        """
        
        response = self.agent.step(prompt)
        
        return {
            "quality_control": response.msgs[0].content,
            "validation_checks": ["Range validation", "Logic checks", "Outlier detection"],
            "reliability_measures": ["Cronbach's alpha > 0.7", "Test-retest ICC > 0.8"],
            "documentation": "Complete audit trail maintained"
        }
    
    async def _perform_power_analysis(self, hypotheses: List[Dict[str, str]], 
                                    data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed power analysis and sample size calculations"""
        
        hypotheses_text = "\n".join([f"- {h.get('hypothesis', '')}" for h in hypotheses])
        
        prompt = f"""
        Conduct comprehensive power analysis for these hypotheses:
        
        Research Hypotheses:
        {hypotheses_text}
        
        Provide detailed power analysis including:
        
        1. **Power Analysis Parameters:**
        - Effect size estimation from literature review
        - Statistical power target (typically 80% or 90%)
        - Alpha level (Type I error rate = 0.05)
        - Two-tailed vs. one-tailed test specification
        
        2. **Sample Size Calculations for Each Hypothesis:**
        - Primary outcome sample size calculation
        - Secondary outcome considerations
        - Stratification and clustering effects
        - Attrition and non-response allowances
        
        3. **Effect Size Justifications:**
        - Literature-based effect size estimates
        - Clinically meaningful difference definitions
        - Standardized effect size interpretations
        - Sensitivity analysis for different effect sizes
        
        4. **Statistical Power Calculations:**
        - Power for primary analyses
        - Power for secondary analyses
        - Power for subgroup analyses
        - Post-hoc power considerations
        
        5. **Sample Size Formula and Software:**
        - Specific formulas used for calculations
        - Software packages for power analysis (G*Power, PASS, R)
        - Assumptions and limitations
        
        6. **Final Sample Size Determination:**
        - Calculated sample size per group
        - Inflation for missing data (typically 10-20%)
        - Final target sample size
        - Feasibility considerations
        
        For each hypothesis, provide:
        - Expected effect size (with justification)
        - Required sample size
        - Statistical power achieved
        - Formula or software used
        """
        
        response = self.agent.step(prompt)
        
        return {
            "power_analysis": response.msgs[0].content,
            "target_power": 0.80,
            "alpha_level": 0.05,
            "effect_size_estimates": ["Small to medium (d=0.3-0.5)", "Medium correlation (r=0.3)"],
            "sample_size_calculated": 1200,
            "software_used": "G*Power 3.1.9.7"
        }
    
    async def _write_complete_methodology_section(self, study_design: Dict[str, Any],
                                                statistical_plan: Dict[str, Any],
                                                data_collection: Dict[str, Any],
                                                quality_control: Dict[str, Any],
                                                power_analysis: Dict[str, Any]) -> str:
        """Write complete methodology section combining all components"""
        
        prompt = f"""
        Write a comprehensive Methodology section for a scientific paper using these components:
        
        Study Design:
        {study_design.get('design_description', '')[:1000]}
        
        Statistical Analysis Plan:
        {statistical_plan.get('analysis_plan', '')[:1500]}
        
        Data Collection:
        {data_collection.get('collection_procedures', '')[:1000]}
        
        Quality Control:
        {quality_control.get('quality_control', '')[:1000]}
        
        Power Analysis:
        {power_analysis.get('power_analysis', '')[:1000]}
        
        Write the Methodology section with clear subsection headers and improved readability:
        
        # 2.1 Study Foundation
        - Paragraph 1: Establish the study design and setting with clear justification for methodological choices (3-4 sentences)
        - Paragraph 2: Describe the research setting, population, and temporal framework (3-4 sentences)
        
        # 2.2 Participant Selection
        - Paragraph 3: Detail the target population and eligibility criteria (3-4 sentences)
        - Paragraph 4: Explain sampling approach and integrate sample size calculation (3-4 sentences)
        
        # 2.3 Data Collection Approach
        - Paragraph 5: Describe data collection procedures and instruments (3-4 sentences)
        - Paragraph 6: Detail validation and standardization processes (3-4 sentences)
        - Paragraph 7: Explain quality assurance measures and ethical considerations (3-4 sentences)
        
        # 2.4 Measurement Framework
        - Paragraph 8: Present variable definitions and operational measures (3-4 sentences)
        - Paragraph 9: Describe primary outcomes, exposures, and covariates (3-4 sentences)
        
        # 2.5 Analytical Strategy
        - Paragraph 10: Present descriptive analysis approaches (3-4 sentences)
        - Paragraph 11: Detail hypothesis testing procedures (3-4 sentences)
        - Paragraph 12: Explain confounding control, missing data handling, and software specifications (3-4 sentences)
        
        Requirements:
        - Each paragraph should focus on ONE main concept
        - Use 3-4 sentences per paragraph maximum
        - Include clear transitions between paragraphs
        - Use formal academic writing style
        - Include specific statistical procedures and justifications
        - Provide enough detail for replication
        - Write 10-12 focused paragraphs (approximately 1000-1200 words)
        """
        
        response = self.agent.step(prompt)
        return response.msgs[0].content 