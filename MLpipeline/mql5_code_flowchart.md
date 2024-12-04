```
    graph TD
    A[Start Script] --> B[Read and Parse JSON File]
    B --> |JSON Parsed Successfully| C[Check Today's Date with JSON Date]
    B --> |JSON Parsing Failed| Z[Exit Script]
    C --> |Dates Match and No Trade Executed| D[Execute Trades from JSON]
    C --> |Dates Don't Match or Trade Already Executed| Z[Exit Script]
    D --> E[Set trade_executed Flag to True]
    E --> F[Record tradeStartTime]
    F --> G[Monitor Time for Trade Closure]
    
    G --> H[Check Current GMT Time]
    H --> |Time >= 20:00 GMT| I[Close All Trades]
    H --> |Time < 20:00 GMT| J[Continue Monitoring]

    I --> Z[Exit Script]
    J --> H[Recheck Time After 60 Seconds]
    
    Z[Exit Script]

    subgraph JSON Data
    B1[Extract Date]
    B2[Extract Symbols and Actions]
    B1 --> C
    B2 --> D
    end

    subgraph Trade Execution Logic
    D1[Loop Through Symbols]
    D2[Execute Buy/Sell Trades]
    D --> D1
    D1 --> D2
    D2 --> E
    end

    subgraph Time Monitoring
    G1[Check Current Time in GMT]
    G2[Compare with 20:00 GMT]
    G --> G1
    G1 --> G2
    end

```
