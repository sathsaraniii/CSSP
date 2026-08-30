# CSSP

```text
                     INPUT MATRIX A
                          │
             ┌────────────┴────────────┐
             │                         │
      ROW-SPACE STAGE           ROW-SPACE STAGE
      Deterministic              Randomized
      Truncated SVD              Randomized Sketch/SVD
             │                         │
             └────────────┬────────────┘
                          │
                 DOMINANT ROW SPACE
                          │
             ┌────────────┴────────────┐
             │                         │
       COLUMN SELECTION          COLUMN SELECTION
         Osinsky (D)               ARP (R)
             │                         │
             └────────────┬────────────┘
                          │
                 SELECTED COLUMNS C
                          │
                   RECONSTRUCTION
                          │
                       METRICS


```
