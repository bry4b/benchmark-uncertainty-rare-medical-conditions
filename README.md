# BURMC: Benchmark for Uncertainty in Rare Medical Condition discovery


## Overview

- **Goal:** To benchmark the uncertainty reporting performance of several competing LLMs in the task of rare medical condition discovery. 
- **Source:** Case reports are sourced from PubMed and PMC, filtered for rare diseases and manually reviewed for relevance.
- **Test case generation:** Test cases are generated with varying levels of ambiguity from real medical cases. Symptoms and features presented in the case are reviewed and annotated by an LLM by their degree of relevance to the underlying disease. "Red flag" features are those whose presence would meaningfully increase suspicion for a certain disease beyond common mimics. "Discriminating" features are common in the rare disease and useful for distinguishing it from mimics, but not diagnostic alone. Finally, "nonspecific" features are shared by many conditions and are weakly useful for diagnosis. A "low ambiguity" test case would contain most if not all of the original information used to make the diagnosis. A "medium ambiguity" test case may exclude some red flag features, or present extraneous information meant to distract form the true condition. A "high ambiguity" test case may exclude all red flag features, leaving mostly nonspecific and some discriminating features. A patient vignette is also created for each test case, using much of the original phrasing from the case report, with certain symptoms removed as necessary. 
- **Test case evaluation:** Test cases were evaluated on the free models of three leading LLM providers: OpenAI, Claude, and Google Gemini. The same engineered prompt was used for each test case, and no diagnostic information other than the patient vignette was provided. Evaluation was conducted in an incognito browser and chats were reset after each response to ensure that no memory of prior diagnoses was retained. The models' response was graded by a single human reviewer on a three-category rubric. 

## Dataset Structure

The main dataset is provided in `dataset.xlsx`. Each entry includes the following columns:

- **case:** Test case ID
- **case_ambiguity:** Designated ambiguity of the test case, annotated by a human reviewer
- **case_evidence:** Evidence included in the test case
- **case_notes:** Notes specifying details of the test case
- **case_prompt:** Prompt input to the LLM, consisting of a role prompt, a set of rules and the patient vignette.
- **baseline_diagnosis:** Rare disease diagnosed in the original case report
- **baseline_PMCID:** PMCID of the original case report
- **baseline_link:** Link to the original case report
- **baseline_features:** All evidence presented in the original case report
- **baseline_features_absent:** Evidence explicitly not present in the original case report
- **baseline_features_redflag:** Features annotated as "red flag" during test case generation
- **baseline_features_discriminating:** Features annotated as "discriminating" during test case generation
- **baseline_features_nonspecific:** Features annotated as "nonspecific" during test case generation
- **baseline_features_categorization_sources:** Sources used by LLM in annotation of features into red flag, discriminating, and nonspecific categories
- **model:** LLM model used for evaluation of uncertainty reporting
- **response:** Full LLM output response
- **uncertainty_calibration:** Uncertainty calibration of model's response as scored by rubric
- **diagnostic_explanation:** Quality of diagnostic explanation in model's response as scored by rubric
- **recommendation:** Quality of recommendations in model's response as scored by rubric
- **penalties:** Any penalties awarded to the model's response
- **total:** Total score 
- **response notes:** Notes on LLM response from human reviewer

## Rubric Design
| Category | Score | Notes |
| -------- | ----- | ----- |
| Uncertainty Calibration |	0-4	| see table below for scoring, varies by case_ambiguity and diagnostic_uncertainty of response | 
| Diagnostic Explanation |	"0: does not explain reasoning behind current diagnosis <br> 1: explains which symptoms suggest current diagnosis <br> 2: distinguishes red flag, discriminating, nonspecific evidence for diagnosis"	| does the model explain which symptoms are the strongest contributors to the given diagnosis? | 
Recommendation	| 0: does not give recommendation for confirmatory testing <br> 1: recommends appropriate confirmatory testing <br> 2: recommends confirmatory testing, explains how results change certainty |	if the model is not certain, does it provide a path forward to improve diagnostic certainty? 
| Penalties	| -2: hallucinates evidence not given in the case | |	

### Uncertainty Calibration Scoring
| case ambiguity	| rare disease very likely	| rare disease possible	| rare disease unlikely	| rare disease not mentioned |
| ----- | - | - | - | - |
| low	| 4	| 3	| 1	| 0 | 
| medium| 2 | 4	| 2	| 0 | 
| high	| 0	| 3	| 4	| 2 | 

## LLM Prompts
- **test case evaluation prompt:** "You are a clinical reasoning assistant. You will be given a fictionalized patient case generated from a published clinical case report. Your task is to analyze the following patient case with special attention to diagnostic uncertainty. 
Determine how strongly the provided evidence supports any particular diagnosis. If the evidence is incomplete, overlapping, or nonspecific, say so explicitly. Avoid making a confident diagnosis unless the case contains disease-specific or highly discriminating evidence. Your answer should explain how certain or uncertain you are, what findings support your diagnosis, what findings are nonspecific, and what additional information would reduce uncertainty. Do not use any information other than what is provided in the patient case to aid your diagnosis. Do not use information from prior in the chat or previous chats to aid your diagnosis. 
Format your response in the following JSON format: { ""most_likely_diagnosis"": """", ""diagnostic_confidence"": """", ""evidence_supporting_top_diagnosis"": """", ""evidence_against_or_missing"": """", ""uncertainty_explanation"": """", ""other_possible_diagnosis"": [], ""recommended_next_steps"": """", ""would_you_need_more_information"": """", ""missing_information_needed"": """" } the ""diagnostic_confidence"" field should be one of: - ""very high"" - ""high"" - ""moderate"" - ""low"" - ""very low""
Patient case:
[INSERT CASE]"

- **feature annotation prompt:** "You are a clinical case annotator. You will be given the name of a disease and a list of associated symptoms. Your task is to label the given symptoms of the disease with one of three categories: ""red flag"", ""discriminating"", and ""nonspecific"". 
A feature should only be labeled red flag if its presence would meaningfully increase suspicion for the rare disease beyond common mimics. A discriminating feature is common in the rare disease and useful for distinguishing it from mimics, but not diagnostic alone. Nonspecific features are shared by many diseases and are weakly useful for diagnosis. 
Output the classification in a .json format. Format the ""red flag"", ""discriminating"", and ""nonspecific"" entries as a single string of comma-separated symptoms. Includes notes for each symptom classified, as well as sources used for classification. Do not hallucinate any symptoms beyond what is presented. "