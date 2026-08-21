# Eval_RAG

How to run code 

python foundry_rag_evaluator.py ^
  --ground-truth ground_truth.json ^
  --responses response_generator.json ^
  --output evaluation_foundry_result.json ^
  --excel-output evaluation_foundry_result.xlsx ^
  --raw-output foundry_raw_output.json



  To include To include XPIA:

  python foundry_rag_evaluator.py ^
  --ground-truth ground_truth.json ^
  --responses response_generator.json ^
  --metrics retrieval groundedness relevance response_completeness hate_unfairness ungrounded_attributes indirect_attack


  Ground Truth JSON                RAG Response JSON
       │                               │
       └─────────────┬─────────────────┘
                     ↓
              Input validation
                     │
                     ↓
                Stable case_id
                     │
          ┌──────────┴──────────┐
          │                     │
          ↓                     ↓
 LOCAL RETRIEVAL            FOUNDRY CLOUD
  EVALUATION                 EVALUATION
          │                     │
 Context Recall              Retrieval
 Context Precision           Groundedness
 Hit@K                       Relevance
 MRR                         Completeness
                             Safety
          │                     │
          └──────────┬──────────┘
                     ↓
              Result normalizer
                     │
           ┌─────────┴─────────┐
           ↓                   ↓
       PASS / FAIL       MISSING / N/A
           │                   │
           └─────────┬─────────┘
                     ↓
                Hard Gates
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       JSON report          Excel report
          │
          ↓
        CI/CD
