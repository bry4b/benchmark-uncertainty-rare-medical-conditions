# Rare Disease Diagnosis Uncertainty Benchmark | Project Plan

Goal: Benchmark how well AI models express unccertainty when diagnosing rare diseases from a possibly ambiguous set of symptoms. 

Dataset building steps: 
1. Pull clinical case reports of rare diseases from PubMed. 
2. Extract symptoms and test results to form a model patient case of the rare disease. 
3. Review symptoms and test results for accuracy and categorize into disease representativeness.
    - Do not trust extracted symptoms unless supported by evidence. 
    - Some features may strongly point toward the rare disease, while others are shared with many diseases.
    - For rare diseases, Orphanet contains symptom frequency. For mimic diseases, clinical references or review papers may be used if symptom frequency is desired. 
4. Generate the following cases. 
    - A low ambiguity case should contain most of the original clinically important information. 
    - A mild ambiguity case should remove labs or tests that confirm the diagnosis. 
    - A moderate ambiguity case should remove rare-disease-specific features. 
    - A high ambiguity case should keep only nonspecific symptoms. 

Rules: 
As an AI agent, you can read any files. You may NOT delete any files. You may write drafts of new files, after which you should ask me for approval before continuing. If you want to make edits to existing files, make your edits in a copy of the file and ask me for approval before continuing.  

