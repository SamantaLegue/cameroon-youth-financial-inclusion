Financial Inclusion of Out-of-School and NEET Youth in Cameroon

An end-to-end data-analysis and data-storytelling portfolio project by Samanta Ndetio Legue, Data Analyst based in Douala, Cameroon.


Portfolio goal: demonstrate how a data analyst can move from a real financial-sector question to a reproducible analytical workflow, an interpretable model, an interactive website, and practical recommendations for youth-focused financial inclusion.

Founder: Samanta Ndetio Legue · GitHub
Location: Douala, Cameroon · Email: leguesamantha8@gmail.com · Phone: +237 678 007 486

Live demo

Open the current website preview · View the analyst profile

This is a managed preview URL and may not be permanent. Replace it with the permanent deployment URL after publication.

Why this project matters

Out-of-school and NEET youth are not one uniform population. Some work informally, search for jobs, operate small businesses, receive transfers, or support their households. They may still face barriers to formal accounts, mobile money, digital payments, savings, and safe financial participation.

The project examines financial inclusion as more than account ownership. It asks whether services are affordable, trusted, reachable, understandable, and useful for young people with irregular income or limited access to formal institutions.

Questions addressed

The analysis and website are organized around four questions:

1.
How can account ownership and meaningful financial use be measured separately?

2.
Which barriers are most relevant, including cost, distance, documentation, trust, connectivity, and digital capability?

3.
Which characteristics are associated with mobile-money usage in the available data?

4.
What could financial institutions, fintechs, government, and youth organizations test in a Cameroon pilot?

What I built

Area
Portfolio evidence
Data analysis
Data cleaning, indicator construction, missing-value review, subgroup comparisons, and barrier analysis
Exploratory visualization
Distribution charts, barrier summaries, correlation analysis, and model feature-importance charts
Machine learning
A Random Forest classifier exploring factors associated with the mobile-money-user indicator
Responsible interpretation
Explicit separation of national, regional, adult-level, and youth-specific evidence
Product communication
An interactive React website with charts, filters, resources, and stakeholder calls to action
AI integration
A server-side educational assistant focused on financial inclusion for out-of-school youth
Stakeholder communication
Policy recommendations, presentation materials, interview pitch, and Q&A preparation




Analytical workflow

Plain Text


Define the problem
        ↓
Document indicators and target population
        ↓
Clean and standardize the source data
        ↓
Explore access, usage, barriers, and subgroup patterns
        ↓
Train and evaluate an exploratory Random Forest model
        ↓
Translate findings into product and policy questions
        ↓
Communicate the evidence through the interactive website



Exploratory data analysis

The Python workflow creates descriptive summaries, missingness information, barrier-rate outputs, numeric correlations, and presentation-ready PNG charts. It accepts a tidy CSV using the fields documented in data_dictionary_out_of_school_youth.csv. Common World Bank-style aliases such as country_name, region_name, and survey_year are also recognized.

Random Forest model

The model uses mobile_money_user as the exploratory target. If that field is not present, the script derives it from mobile_money_ownership and digital_payment when those fields exist. Candidate predictors include account ownership, digital payment, phone access, internet access, age, income group, location, gender, region, and year.

The model uses a preprocessing pipeline with numeric imputation, categorical imputation, one-hot encoding, and a class-balanced Random Forest classifier. It writes held-out metrics, a classification report, impurity-based feature importance, permutation importance, and a feature-importance chart.

Important interpretation boundary: the model is a portfolio demonstration of supervised-learning workflow and feature interpretation. It must not be used to approve or deny accounts, rank young people, infer individual risk, or replace human review. If the input is country-year aggregate data rather than respondent-level data, the model describes associations between aggregate indicators and cannot be interpreted as an individual prediction model.

Data and limitations

The project was developed around World Bank Global Findex indicators and related financial-inclusion evidence. The exact source file used for any published result must be retained with its license, year, population, unit, denominator, and transformation notes.

The website includes filters for Centre, Littoral, North-West, and Far North. Those regional views are intentionally marked Data pending until verified and comparable Cameroon regional data is connected. The interface-ready filters are not regional findings.

Adult-level mobile-money benchmarks must not be presented as youth-specific outcomes. A future pilot should collect or connect verified youth-level survey or administrative data, document the sampling design, and validate indicators across the four regions.

The website’s AI assistant is educational. Users should not enter personal, identity, account, or financial information. The assistant does not provide financial advice and does not replace verified statistics or professional review.

Repository structure

Plain Text


.
├── analysis/
│   └── eda_random_forest.py       # EDA and Random Forest workflow
├── client/                        # React frontend
├── server/                        # Express/tRPC backend and AI procedure
├── shared/                        # Shared types and constants
├── drizzle/                       # Database schema and migrations
├── data_dictionary_out_of_school_youth.csv
├── financial_inclusion_report.md
├── cameroon_policy_recommendations.md
├── presentation/                  # Interview pitch outline
├── presentation_outline_and_qa.md
├── full_presentation_script_with_website_cues.md
└── README.md



Run the Python analysis

The public repository should contain only data that is legally shareable and permitted by the source license. Place an authorized CSV in a local path such as data/processed/financial_inclusion.csv.

Create a Python environment and install the analysis dependencies:

Bash


python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn



Run the workflow:

Bash


python analysis/eda_random_forest.py \
  --input data/processed/financial_inclusion.csv \
  --output outputs/eda



The script writes descriptive_summary.csv, barrier_rates.csv, model_metrics.json, feature-importance CSV files, PNG charts, and run_summary.json to the selected output directory. Generated outputs should remain local unless they are reviewed and intentionally added to the portfolio.

Run the website locally

The website uses Node.js, pnpm, React, Vite, Express, tRPC, TypeScript, and a managed server-side AI integration.

Bash


pnpm install
pnpm check
pnpm test
pnpm build
pnpm dev



The full-stack environment requires the approved project configuration for the database, authentication, storage, and server-side AI integration. Never commit .env files, API keys, session secrets, database credentials, private data, or local logs.

Portfolio presentation

When presenting the project to an interviewer, use this sequence:

1.
State the financial-sector problem and why out-of-school youth are a meaningful population.

2.
Define the analytical questions and distinguish account access from useful, safe usage.

3.
Show the data workflow, including cleaning, indicator design, EDA, and modeling.

4.
Demonstrate the website’s insights, Cameroon filters, responsible Data pending state, and AI assistant.

5.
Close with the practical value: a repeatable framework for designing and evaluating a youth financial-inclusion pilot.

The interview-focused outline is available in presentation/interviewer_pitch_outline.md.

Next phase and call to action

The next analytical step is to connect verified Cameroon regional data and test the framework with local partners. Samanta is open to conversations with financial institutions, fintech companies, youth organizations, researchers, and public-sector stakeholders interested in funding a youth financial-inclusion pilot, supporting a pilot survey in Douala, sharing verified regional data, or joining as a research partner.

References

[1] World Bank Global Findex Database
[2] UNCDF, Cameroon: Next steps in a high-potential market for Digital Financial Services
[3] World Bank Global Findex, The impact of mobile money in Sub-Saharan Africa
This project is intended for portfolio demonstration, learning, and responsible pilot design. Validate current local statistics before using the work for operational, financial, or policy decisions.

